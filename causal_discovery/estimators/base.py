import datetime
import time

import numpy as np
import pandas as pd


class base_discover():
    def __init__(self, dataset, spark_context=None):
        self._dataset = dataset
        self._spark_context = spark_context
        print("Data size: {}".format(dataset.data.shape[0]))
        
    @staticmethod
    def _get_empty_edge_table():
        edge_table = pd.DataFrame(columns=[
            'method',
            'edge_start',
            'edge_end',
            'estimate_value'
        ])
        return edge_table

    @staticmethod
    def _update_edge_table(edge_table,
                          method_name,
                          edge_start,
                          edge_end,
                          estimate_value):
        
        # edge_table.loc[len(edge_table)] = [
        #     method_name, edge_start, edge_end, estimate_value
        # ]
        edge_table['method'] = method_name
        edge_table['edge_start'] = edge_start
        edge_table['edge_end'] = edge_end
        edge_table['estimate_value'] = estimate_value



    def _get_edge_table(self,
                       method_name,
                       is_sparse=False,
                       edge_data_map=None,
                       weights=None):

        start_time = time.time()

        edge_table = self._get_empty_edge_table()

        if is_sparse is False:
            n = edge_data_map.shape[0]
            edge_start_list = []
            edge_end_list = []
            estimate_value_list = []
            method_name_list = []

            for i in range(n):
                for j in range(n):
                    if edge_data_map[i][j] != 0:
                        estimate_value_list.append(edge_data_map[i][j])
                        edge_start_list.append(i)
                        edge_end_list.append(j)
                        method_name_list.append(method_name)
        else:
            method_name_list = [method_name] * edge_data_map.shape[0]
            edge_start_list = edge_data_map.iloc[:,0]
            edge_end_list = edge_data_map.iloc[:,1]
            estimate_value_list = edge_data_map.iloc[:,2]

        self._update_edge_table(edge_table, method_name_list, edge_start_list, edge_end_list, estimate_value_list)

        return edge_table

        
        
        
        
        
        
        
        
