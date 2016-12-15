
from scipy import stats
import numpy as np
import collections


class DataFrameOps(object):

    def __init__(self):
        pass

    def _get_diff_by_id(self, roundA, roundB, field_id):

        # Get these two rounds
        dataA = self._get_field_by_id(roundA, field_id)
        dataB = self._get_field_by_id(roundB, field_id)

        # Now take the difference of the rounds and drop any that have data missing
        return (dataB - dataA).dropna()

    def _get_diff_by_name(self, roundA, roundB, field_name):

        # Get these two rounds
        dataA = self._get_field_by_name(roundA, field_name)
        dataB = self._get_field_by_name(roundB, field_name)

        # Now take the difference of the rounds and drop any that have data missing
        return (dataB - dataA).dropna()

    def _get_percent_by_name(self, roundA, roundB, field_name):

         # Get these two rounds
        dataA = self._get_field_by_name(roundA, field_name)
        dataB = self._get_field_by_name(roundB, field_name)

        return ((dataB - dataA) / dataA).dropna()

    def _get_diff_participant(self, roundA, roundB, username):

         # Get these two rounds
        dataA = self._get_participant_by_name(roundA, username)
        dataB = self._get_participant_by_name(roundB, username)

        # Now take the difference of the rounds and drop any that have data missing
        return (dataB - dataA).dropna()

    def _get_all_diff_participant(self, roundA, roundB):

        # Get these two rounds
        dataA = self._get_all_participants(roundA)
        dataB = self._get_all_participants(roundB)

        # Now take the difference of the rounds and drop any that have data missing
        return (dataB - dataA)

    def _get_all_diff(self, roundA, roundB):

        # Get these two rounds
        dataA = self._get_all_fields(roundA)
        dataB = self._get_all_fields(roundB)

        # Now take the difference of the rounds and drop any that have data missing
        return (dataB - dataA)

    def _get_signrank_by_name(self, roundA, roundB, field_name):

        r1 = self._get_field_by_name(roundA, field_name)
        r2 = self._get_field_by_name(roundB, field_name)

        # Merge the two in a dataframe
        data = r1.join(r2, lsuffix='_r1', rsuffix='_r2')

        # Drop rows with NaN values
        data.dropna()

        # Separate out the data
        d1 = data[field_name+'_r1']
        d2 = data[field_name+'_r2']

        z_stat, p_val = stats.ranksums(d1, d2)
        return (z_stat, p_val, np.mean(d1), np.mean(d2), np.std(d1), np.std(d2))

    def _username_round_index(self, dataframe, username_col='username', round_col='round', drop=True):
        """
        Given a dataframe with username and round columns return 
        a new dataframe with a unique index based on them
        """
        dataframe = dataframe.copy()
        if round_col in dataframe.columns:
            dataframe['tmp'] = dataframe.apply( lambda row: '%s_%s' % (row[username_col], row[round_col]), axis=1)
            dataframe = dataframe.set_index('tmp')
            if drop:
                dataframe = dataframe.drop(username_col, 1)
                dataframe = dataframe.drop(round_col, 1)
        else:
            dataframe = dataframe.set_index(username_col, drop=drop)
        return dataframe 

    def _map_uname_rnd(self, row):
        #print row
        return "%s_%i" % (row['username'], row['round'])

    def min_fill(self, dataframe, adjust=.1):
        """
        Assumes dataframe is vectorized or correlized,
        """
        return dataframe.fillna( dataframe.min() * adjust )

    def get_entropy(self, dataframe, axis=0):
        def _get_ent(vec):
            seq = [ x for x in collections.Counter(vec).itervalues() ]
            length = float(len(vec))
            pk = [v/length for v in seq]
            return stats.entropy( pk )
        return dataframe.apply( _get_ent, axis=0)
