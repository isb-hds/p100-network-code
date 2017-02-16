import statsmodels.sandbox.stats.multicomp
import scipy.stats
import pandas
from collections import Counter
import itertools
import time
import numpy as np
import numpy.linalg.linalg
import networkx as nx
from datasource  import DataSourceFactory
import statsmodels.regression.mixed_linear_model as mlm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import os
import logging
from dataframeops import DataFrameOps

l_logger = logging.getLogger("p100.utils.correlations.analysis")

class Analysis:

    def __init__(self, ds_id_map, part_df, data_dir):

        self._user_indices = None
        self._datasources = {}
        self._debug = {}
        self._correlation_results = []
        #cache used by parallel to avoid reloading dataframes between messages
        self._all_tests = [self.mixed_effects, self.spearman, self.kruskal, self.kendalltau, self.kruskal]
        l_logger.info("Initializing Analysis %s" % (self))
        self.ds_map = pandas.read_pickle(ds_id_map)
        self.part_df = part_df
        self.ds_id_map = ds_id_map
        self.data_dir = data_dir


    def GetResult(self, annotated=False,
                  entropy=False, datasource=None, disjoint=False,
                  nocache=False):

        result = pandas.DataFrame(self._correlation_results, columns=[
                    'ds_id_1', 'ds_id_2','variable_id_1', 'variable_id_2',
                     'test', 'coefficient', 'pval', 'pval_adj']
            )
        if disjoint:
            result = result[result.ds_id_1 != result.ds_id_2]
        if annotated and len(result.index) > 0:
            l_logger.debug("annotating")
            result = self.add_annotations(result)
        if entropy:
            l_logger.debug("Annotating variables with their entropy.")
            result = self.add_entropy(result)
        return result

    def GetDataSources(self, type=None):
        ds_ids = self.ds_map.ds_id.unique().tolist()
        dsf = DataSourceFactory(ds_id_map = self.ds_id_map, part_df=self.part_df, data_dir=self.data_dir)
        ds = []
        for ds_id in ds_ids:
            ds_obj = dsf.get_by_ds_id(ds_id)
            if type is None or ds_obj.type == type:
                ds.append(ds_obj)
        return ds

    def to_graph(self, dataframe):
        G = nx.Graph()
        for i, row in dataframe.iterrows():
            G.add_node(
                row.annotations_1, ds=row.annotations_1.split('.')[0])
            G.add_node(
                row.annotations_2, ds=row.annotations_2.split('.')[0])
            G.add_edge(
                row.annotations_1, row.annotations_2, coef=row.coefficient)
        return G

    def add_annotations(self, dataframe):
        dsf = DataSourceFactory(ds_id_map = self.ds_id_map, part_df=self.part_df, data_dir=self.data_dir)
        for ds_id in dataframe.ds_id_1.unique():
            sel = (dataframe.ds_id_1 == ds_id)
            anno_df = dsf.get_by_ds_id(ds_id).annotations
            anno_list = anno_df.loc[dataframe[sel]['variable_id_1']].tolist()
            dataframe.loc[sel, 'annotations_1'] = anno_list
            l_logger.debug("%i: annotated" % ds_id)
        for ds_id in dataframe.ds_id_2.unique():
            sel = (dataframe.ds_id_2 == ds_id)
            anno_df = dsf.get_by_ds_id(ds_id).annotations
            anno_list = anno_df.loc[dataframe[sel]['variable_id_2']].tolist()
            dataframe.loc[sel, 'annotations_2'] = anno_list
            l_logger.debug("%i: annotated" % ds_id)
        return dataframe

    def add_entropy(self, dataframe):
        dsf = DataSourceFactory(ds_id_map = self.ds_id_map, part_df=self.part_df, data_dir=self.data_dir)
        dfo = DataFrameOps()
        cache = {}
        for ds_id in dataframe.ds_id_1.unique():
            cache[ds_id] = dfo.get_entropy(
                dsf.get_by_ds_id(ds_id).GetDataFrame())
        for ds_id in dataframe.ds_id_2.unique():
            if ds_id not in cache:
                cache[ds_id] = dfo.get_entropy(
                    dsf.get_by_ds_id(ds_id).GetDataFrame())
        for ds_id in cache.keys():
            sel_1 = (dataframe.ds_id_1 == ds_id)
            dataframe.loc[dataframe[sel_1].index, 'entropy_1'] = \
                cache[ds_id][dataframe[sel_1].variable_id_1].tolist()
            sel_2 = (dataframe.ds_id_2 == ds_id)
            dataframe.loc[dataframe[sel_2].index, 'entropy_2'] = \
                cache[ds_id][dataframe[sel_2].variable_id_2].tolist()
        return dataframe



    def Correlate(self, datasource1, datasource2, tests=None,
                  save=True, delta=False, cutoff=.1,mean=False, mean_age_sex=True, delta_age_sex=False):
        start = time.time()
        l_logger.info("Correlating %s with %s" %
                      (str(datasource1), str(datasource2)))
        ignored = 0
        if tests is None:
            # self.tests = [self.spearman, self.pearson,
            # self.kendalltau, self.kruskal]
            self.tests = [self.spearman, self.kruskal]
        else:
            self.tests = tests
        if delta:
            df1 = datasource1.delta_transform()
            df2 = datasource2.delta_transform()
        elif mean:
            df1 = datasource1.mean_transform()
            df2 = datasource2.mean_transform()
        elif mean_age_sex:
            df1 = datasource1.mean_transform_agesex_adjust()
            df2 = datasource2.mean_transform_agesex_adjust()
        elif delta_age_sex:
            df1 = datasource1.delta_transform_agesex_adjust()
            df2 = datasource2.delta_transform_agesex_adjust()
        else:
            df1 = datasource1.GetDataFrame()
            df2 = datasource2.GetDataFrame()

        correlations = []
        tests = []
        ctr = 1
        n = float(len(df1.columns) * len(df2.columns))
        l_logger.debug("%.2e comparisons" % (n,))
        # a list of lists of pvalue pvs[0] is pvals from self.tests[0]
        pvs = []
        # a list of lists of coeff coeffs[0] is coeffs from self.tests[0]
        coeffs = []
        test_iter = [(i, test['func'], list(), list())
                     for i, test in enumerate(self.tests)]
        ds_id_1 = datasource1.id
        ds_id_2 = datasource2.id
        corr_header = (ds_id_1, ds_id_2)
        for col1, col2 in itertools.product(df1.columns, df2.columns):

            if ds_id_1 == ds_id_2 and col1 >= col2:
                # if we are running an intra data source comparison
                # (i.e. chem to chem)
                # then only compare a pair of variables once
                continue
            c1 = "a.%s" % col1
            c2 = "b.%s" % col2

            merged = pandas.concat(
                [df1[col1], df2[col2]], axis=1).dropna(axis=0)
            if len(merged) > 10:  # check enough observations to test

                try:
                    merged.columns = [c1, c2]
                except ValueError:
                    return (merged, df1, df2)
                ser1, ser2 = merged[c1], merged[c2]
                correlations.append((col1, col2))
                for i, f, pv, co in test_iter:
                    coeff, pvalue = f(ser1, ser2)
                    pv.append(pvalue)
                    co.append(coeff)
                ctr += 1
            else:
                ignored += 1
            if n > 10 and ctr % int(n / 10) == 0:
                l_logger.info("%i percent done" %
                              (int((ctr / n) * 100)))
        for _, _, pv, co in test_iter:
            pvs.append(pv)
            coeffs.append(co)

        if len(correlations) > 0 and save:
            corr_df = pandas.DataFrame(
                correlations, columns=['ds_id_1',  'ds_id_2'])
            coeff_df = pandas.DataFrame(
                coeffs, index=[t['label'] for t in self.tests]).transpose()
            pv_df = pandas.DataFrame(
                pvs, index=[t['label'] for t in self.tests]).transpose()
            pv_adj = pv_df.copy().fillna(1.0)
            l_logger.debug("returning pv_adj")
            for c in pv_df.columns:
                pv_adj[c] = self.correct(pv_df[c])
            pv_adj = pv_adj.fillna(1.0)
            l_logger.debug("%s and %s correlations complete[%i comparisons ignored]" % (
                str(datasource1), str(datasource2), ignored))
            l_logger.debug("xx - %.1f" % (time.time() - start,))
            self._save(
                corr_header, corr_df, coeff_df, pv_df, pv_adj, cutoff=cutoff)
            l_logger.debug("zz - %.1f" % (time.time() - start,))
        if len(correlations) == 0:
            l_logger.warning(
                "No comparisons for %s v %s" % (str(datasource1), str(datasource2)))

    def _save(self, corr_header, correlations, coefficients, pv, pv_adj,  cutoff=.1):
        df_mask = np.any(pv_adj <= cutoff, axis=1)
        correlations = correlations[df_mask]
        coefficients = coefficients[df_mask]
        pv = pv[df_mask]
        pv_adj = pv_adj[df_mask]
        res = []
        if len(correlations) > 0:
            l_logger.info("Saving correlations")

            l_logger.debug("Getting corr_relation dataframe")
            for idx, row in correlations.iterrows():
                for t in coefficients.columns:
                    _ = coefficients.loc[idx, t]
                    _ = pv.loc[idx, t]
                    _ = pv_adj.loc[idx, t]
                    res.append(
                         corr_header + tuple(row) + 
                            ( t, coefficients.loc[idx, t], pv.loc[idx, t],
                         pv_adj.loc[idx, t] ))
            l_logger.info("Saved correlations")
            self._correlation_results += res
        else:
            l_logger.warning(
                "**********There were no correlations after adjustment*******")

    def correct(self, pvals, method='fdr_bh'):
        """
        Given a series of pvalues, correct for multiple hypothesis

        Returns series with same index and corrected values
        """

        pvals_sub = pvals.dropna()
        cutoff = .05
        mt = statsmodels.sandbox.stats.multicomp.multipletests
        pvals_sub = pvals.dropna()
        if len(pvals_sub) > 0:
            (accepted, corrected, unused1, unused2) = mt(
                pvals_sub, method=method, alpha=.05)
            corrSer = pandas.Series(corrected, index=pvals_sub.index)
            return corrSer[pvals.index].fillna(1.0)
        else:
            # all bad
            empty = pvals.copy()
            empty = 1
            return empty

    def accept(self, corrected_pvals, cutoff=.05):
        """
        Given a series of pvalues, return binary vector that selects
        those values under the cutoff
        """
        return (corrected_pvals < cutoff)

    def annotate(self, ds_id, var_id):
        ds_id = int(ds_id)
        var_id = int(var_id)
        if ds_id not in self._datasources:
            dsf = DataSourceFactory()
            self._datasources[ds_id] = dsf.get_by_ds_id(ds_id)
        ds = self._datasources[ds_id]
        return "%s.%s.%s" % (ds.type, str(ds.aux), ds.annotate(var_id))

    def _align(self, ser1, ser2):
        """
        Given 2 column names, return 2 intersected series
        """
        df1_temp = ser1.dropna()
        df2_temp = ser2.dropna()
        df1 = df1_temp[df2_temp.index].dropna()
        df2 = df2_temp[df1.index]
        return (df1, df2)

    @property
    def spearman(self):
        test = {'label': 'SPEARMAN',
                'func': scipy.stats.spearmanr
                }
        return test

    @property
    def pearson(self):
        test = {'label': 'PEARSON',
                'func': scipy.stats.pearsonr
                }
        return test

    @property
    def kendalltau(self):
        test = {'label': 'KENDALL',
                'func': scipy.stats.kendalltau
                }
        return test

    def _kruskal_comp(self, ser1, ser2):
        """
        If the series can be binarized into sets larger than 5, run kruskal.
        """
        comp = []
        for k, v in Counter(ser1.values).iteritems():
            if v > 10:
                comp.append(ser2[ser1[ser1 == k].index])
        if 6 > len(comp) > 1:
            try:
                return scipy.stats.kruskal(*comp)
            except ValueError as v:
                if v.message != 'All numbers are identical in kruskal':
                    raise v
        return (None, 1.0)

    def _kruskal_filter(self, ser1, ser2):
        """
        Runs kruskal on both possible orderings and returns best
        """
        Hstat1, p1 = self._kruskal_comp(ser1, ser2)
        Hstat2, p2 = self._kruskal_comp(ser2, ser1)
        if Hstat1 is None and Hstat2 is None:
            return (None, None)
        if p1 < p2:
            return (Hstat1, p1)
        else:
            return (Hstat2, p2)

    @property
    def kruskal(self):
        test = {'label': 'KRUSKAL',
                'func': self._kruskal_filter}
        return test

    def _mixed_effects_comp( self, ser1, ser2):
        groups = [self.user_indices[i.split('_')[0]] for i in ser1.index.tolist()]
        exo = ser1.values
        endo = ser2.values
        try:
            model = mlm.MixedLM( np.array( endo ),
                                 np.array( exo ),
                                 np.array( groups ) )
            results = model.fit()
            summary = results.summary()
            l_logger.debug( "Summary %r" % ( summary, ) )
            if 'x1' in summary.tables[1].index:
                coeff = float(summary.tables[1].loc[ 'x1', 'Coef.'])
                pv = results.pvalues[0]
            else:
                l_logger.warning("no result")
                #this happens when a vector has 0 entropy
                coeff = None
                pv = 1.0
        except:
            coeff = None
            pv = 1.0
            l_logger.exception("Error computing mixed model")
            raise
        return (coeff, pv)

    def _mixed_effects( self, ser1, ser2 ):
        try:
            coeff1, p1 = self._mixed_effects_comp(ser1, ser2)
        except numpy.linalg.linalg.LinAlgError:
            (coeff1, p1) = None, 1.0
            l_logger.exception("Singular matrix")
        except ConvergenceWarning:
            (coeff1, p1) = None, 1.0
            l_logger.exception("Convergence Error on 1")

        if coeff1 is None or abs(coeff1) < .01:
            coeff1 = None
            p1 = 1.0
        try:
            coeff2, p2 = self._mixed_effects_comp(ser2, ser1)
        except numpy.linalg.linalg.LinAlgError:
            l_logger.exception("Singular matrix")
            (coeff2, p2) = None, 1.0
        except ConvergenceWarning:
            (coeff2, p2) = None, 1.0
            l_logger.exception("Convergence Error on 2")
        if coeff2 is None or abs(coeff2) < .01:
            coeff2 = None
            p2 = 1.0
        if coeff1 is None and coeff2 is None:
            return (None, None)
        if p1 < p2:
            return (coeff1, p1)
        else:
            return (coeff2, p2)

    @property
    def mixed_effects( self ):
        test = {'label': 'MIXED',
                'func': self._mixed_effects
                }
        return test

    @property
    def user_indices( self ):
        if self._user_indices is None:
            self._user_indices = self.getUserIndices()
        return self._user_indices

    def getUserIndices(self):
        part_df = pandas.read_pickle(self.part_df)
        return {k:v for v,k in enumerate(sorted(part_df.username.tolist()))}
