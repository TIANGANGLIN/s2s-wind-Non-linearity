import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pandas as pd
from downscaling.data.GetData import convertToBCWH

class MyNormalizer():
    """
    A custom normalizer class that implements various normalization strategies
    for multi-dimensional meteorological data.
    """
    def __init__(self, standardize, model_name):
        """
        Initialize the normalizer with specific standardization strategy.
        
        Args:
            standardize (str): The normalization strategy to use
            model_name (str): Name of the model for reference
        """
        super().__init__()
        self.standardize = standardize
        self.model_name = model_name

    def fit(self, refer):
        """
        Fit the normalizer by computing statistics from reference data.
        
        Args:
            refer: Reference data to calculate normalization statistics
        """
        if len(refer.dims)!=4:
            de = 0
        refer = refer.data
        # # Assuming X has shape (sample, channels, width, height)
        # Modify standardization strategy to consider both time and spatial dimensions
        if self.standardize == 'Norm0Spatial':
            # Calculate mean and std for each spatial location
            self.spatial_mean = refer.mean(axis=0, keepdims=True)  # Preserve spatial distribution features
            self.spatial_std = refer.std(axis=0, keepdims=True)
            
            # Calculate global mean and std for overall normalization
            self.global_mean = refer.mean()
            self.global_std = refer.std()
        elif self.standardize == 'Norm0SpatialV2':
            # Calculate mean and std for each spatial location
            self.spatial_mean = refer.mean(axis=0, keepdims=True)  # Preserve spatial distribution features
            self.spatial_std = refer.std(axis=0, keepdims=True)
            
            refer_norm = (refer - self.spatial_mean)/self.spatial_std
            # Calculate global mean and std for normalized data
            self.global_mean = refer_norm.mean()
            self.global_std = refer_norm.std()

        elif self.standardize == 'NormSpatioTemporal':
            # Calculate statistics for each spatial location
            self.spatial_mean = refer.mean(axis=0, keepdims=True)
            self.spatial_std = refer.std(axis=0, keepdims=True)
            
            # Calculate statistics across time dimension
            self.temporal_mean = refer.mean(axis=(0, 2,3), keepdims=True)
            self.temporal_std = refer.std(axis=(0, 2,3), keepdims=True)
        elif self.standardize in ['Norm023','Norm023ThenSeason','Norm023OnlyX','Norm023OnlyXThenSeason']:
            # Normalize across sample, height and width dimensions
            self.mean = refer.mean(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            self.std = refer.std(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        elif self.standardize in ['Norm0','Norm0ThenSeason','Norm0OnlyX','Norm0OnlyXThenSeason']:
            # Normalize across sample dimension only
            self.mean = refer.mean(axis=(0, )).reshape(1, *refer.shape[1:])
            self.std = refer.std(axis=(0, )).reshape(1, *refer.shape[1:])
        elif self.standardize in ['Norm023stdSpatial']:
            # Mixed approach: mean across all dimensions, std only across spatial
            self.mean = refer.mean(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            self.std = refer.std(axis=(0, )).reshape(1, *refer.shape[1:])
        elif self.standardize in ['NormGridMeanGlobalStd']:
            # Mixed approach: mean per grid point, global std
            self.mean = refer.mean(axis=(0, )).reshape(1, *refer.shape[1:])
            self.std = refer.std(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        else:
            raise ValueError('Error: Unsupported standardization method')
        
        # if len(self.mean.shape)!=4:
        #     de = 0
        return 

    def transform(self, x):
        """
        Transform input data using the fitted normalization parameters.
        
        Args:
            x: Input data to normalize
        
        Returns:
            Normalized data
        """
        if self.standardize in ['Norm0Spatial','Norm0SpatialV2']:
            # First perform spatial normalization
            x_spatial = (x - self.spatial_mean) / self.spatial_std
            # Then perform global normalization
            x_normalized = (x_spatial - self.global_mean) / self.global_std
            return x_normalized
        elif self.standardize == 'NormSpatioTemporal':
            # First normalize across spatial dimensions
            x_spatial = (x - self.spatial_mean) / self.spatial_std
            
            # Then normalize across temporal dimension
            x_temporal = (x_spatial - self.temporal_mean) / self.temporal_std
            return x_temporal
        else:
            # Standard normalization
            x = (x - self.mean) / self.std
            return x
    
    def inverse_transform(self, x):
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            x: Normalized data
        
        Returns:
            Data in original scale
        """
        if self.standardize in ['Norm0Spatial','Norm0SpatialV2']:
            # Inverse of global normalization
            x = x * self.global_std + self.global_mean
            # Inverse of spatial normalization
            x = x * self.spatial_std + self.spatial_mean
        else:
            # Standard inverse normalization
            x = x * self.std + self.mean

        return x
    
class Distribution():
    """
    A class for applying signed logarithmic transformation to better handle
    skewed distributions and extreme values.
    """
    def __init__(self, ):
        super().__init__()
    
    def transform(self, x):
        """
        Apply signed logarithmic transformation: sign(x) * log(1 + |x|)
        
        Args:
            x: Input data
            
        Returns:
            Transformed data
        """
        return np.sign(x) * np.log(1 + np.abs(x))

    def inverse_transform(self, y):
        """
        Inverse of signed logarithmic transformation: sign(y) * (exp(|y|) - 1)
        
        Args:
            y: Transformed data
            
        Returns:
            Original-scale data
        """
        return np.sign(y) * (np.exp(np.abs(y)) - 1)

class DataProcessor:
    """
    A class for data processing and restoration, including detrending, 
    deseasoning and normalization of meteorological data.
    """
    def __init__(self, args=None, model_name=None):
        """
        Initialize data processor with specific arguments and model.
        
        Args:
            args: Configuration arguments
            model_name: Name of the model
        """
        self.base_date = datetime(1994, 1, 1)
        self.scaler = MyNormalizer(standardize=args.standardize, model_name=model_name)
        self.args = args
        self.seasonal_cycle_test = {}
        
    def _get_relative_days(self, da):
        """
        Calculate days relative to base date.
        
        Args:
            da: Data array containing date information
            
        Returns:
            Array of days relative to base date
        """
        dates = pd.to_datetime(da.N.values)
        relative_days = (dates - self.base_date).days.values.reshape(-1, 1)
        return relative_days

    def fit(self, train_data, train_clim):
        """
        Fit the data processor on training data.
        
        Args:
            train_data: Training data
            train_clim: Climatology data for training period
        """
        train_data = train_data.unstack()
        train_clim = train_clim.unstack().sel(N=train_data.N)
        if self.args.processor in [1, 11]:
            """Fit and transform data with detrending and deseasoning"""
            train_data = train_data.unstack()
            train_clim = train_clim.unstack().sel(N=train_data.N)
            # 1. Save seasonal cycle
            seasonal_cycle_train = train_clim.mean('pdf')
            
            # 2. Remove seasonality
            train_deseasoned = train_data - seasonal_cycle_train
            
            # 3. Fit trend model
            relative_days = self._get_relative_days(train_deseasoned)
            da_tra_flat = (train_deseasoned.mean('pdf')
                        .isel(time=0)
                        .stack(feature=['varOI','lat','lon'])
                        .transpose('N','feature'))
            
            self.detrend_model = LinearRegression()
            self.detrend_model.fit(relative_days, da_tra_flat.values)
            
            # 4. Remove trend
            train_detrended = self._remove_trend(train_deseasoned)
            
            # 5. Fit normalizer
            self.scaler.fit(refer=convertToBCWH(self.args, train_detrended))
        elif self.args.processor in [2, 12]:
            """Fit and transform data with spatial average detrending and deseasoning"""
            train_data = train_data.unstack()
            train_clim = train_clim.unstack().sel(N=train_data.N)
            # 1. Save seasonal cycle (spatial average)
            seasonal_cycle_train = train_clim.mean('pdf').mean(['lat','lon']).expand_dims(lat=train_data.lat.data, lon=train_data.lon.data)
            
            # 2. Remove seasonality
            train_deseasoned = train_data - seasonal_cycle_train
            
            # 3. Fit trend model for spatial average
            relative_days = self._get_relative_days(train_deseasoned)
            da_tra_flat = (train_deseasoned.mean('pdf').mean(['lat','lon']).expand_dims(lat=train_data.lat.data, lon=train_data.lon.data)
                        .isel(time=0)
                        .stack(feature=['varOI','lat','lon'])
                        .transpose('N','feature'))
            
            self.detrend_model = LinearRegression()
            self.detrend_model.fit(relative_days, da_tra_flat.values)
            
            # 4. Remove trend
            train_detrended = self._remove_trend(train_deseasoned)
            
            # 5. Fit normalizer
            self.scaler.fit(refer=convertToBCWH(self.args, train_detrended))
        else:
            # Simple normalization without detrending/deseasoning
            if self.args.fcProcessor in ['subRA', 'RA']:
                # Fit using reanalysis data
                self.scaler.fit(refer=convertToBCWH(self.args, train_data.unstack().mean('pdf').expand_dims(pdf=[0])))
            elif self.args.fcProcessor in ['subCL', 'CL']:
                # Fit using climatology data
                self.scaler.fit(refer=convertToBCWH(self.args, train_clim.unstack().mean('pdf').expand_dims(pdf=[0])))
            else:
                raise ValueError('Invalid fcProcessor option')
        return 

    def transform(self, test_data, test_clim, save_key):
        """
        Transform test data using fitted processor.
        
        Args:
            test_data: Test data to transform
            test_clim: Climatology data for test period
            save_key: Key to save seasonal cycle information
            
        Returns:
            Transformed test data
        """
        test_data = test_data.unstack()
        test_clim = test_clim.unstack().sel(N=test_data.N)
        if self.args.processor in [1, 11]:
            # 1. Save seasonal cycle
            self.seasonal_cycle_test[save_key] = test_clim.mean('pdf').mean(['lat','lon'])
            
            # 2. Remove seasonality
            test_deseasoned = test_data - self.seasonal_cycle_test[save_key]
            
            # 4. Remove trend
            test_detrended = self._remove_trend(test_deseasoned)

            # 6. Normalize
            test_normalized = self.scaler.transform(convertToBCWH(self.args, test_detrended))
        elif self.args.processor in [2, 12]:
            # 1. Save seasonal cycle (spatial average)
            self.seasonal_cycle_test[save_key] = test_clim.mean('pdf').mean(['lat','lon']).expand_dims(lat=test_data.lat.data, lon=test_data.lon.data)
            
            # 2. Remove seasonality
            test_deseasoned = test_data - self.seasonal_cycle_test[save_key]
            
            # 4. Remove trend
            test_detrended = self._remove_trend(test_deseasoned)

            # 6. Normalize
            test_normalized = self.scaler.transform(convertToBCWH(self.args, test_detrended))
        else:
            # Simple normalization
            test_normalized = self.scaler.transform(convertToBCWH(self.args, test_data))

        # Apply distribution transformation if specified
        if self.args.processor in [10, 11, 12]:
            test_normalized = Distribution().transform(test_normalized)

        return test_normalized
        
    def _remove_trend(self, da):
        """
        Remove linear trend from data.
        
        Args:
            da: Data array to detrend
            
        Returns:
            Detrended data array
        """
        relative_days = self._get_relative_days(da)
        da_stacked = da.stack(feature=['varOI','lat','lon']).transpose('N','feature',...)
        trend = self.detrend_model.predict(relative_days)
        
        detrended = xr.zeros_like(da_stacked)
        for pdf in da_stacked.pdf.values:
            for time in da_stacked.time.values:
                trend = self.detrend_model.predict(relative_days)
                detrended.loc[{'pdf':pdf,'time':time}] = (
                    da_stacked.sel(pdf=pdf, time=time) - trend
                )
        
        return detrended.unstack('feature')
    
    def inverse(self, normalized_data, save_key):
        """
        Inverse transform data back to original scale.
        
        Args:
            normalized_data: Normalized data to inverse transform
            save_key: Key to retrieve seasonal cycle information
            
        Returns:
            Data in original scale
        """
        # Inverse transform distribution if applied
        if self.args.processor in [10, 11, 12]:
            normalized_data = Distribution().inverse_transform(normalized_data)
            
        if self.args.processor in [1, 2, 11, 12]:
            # 1. Inverse normalize
            denormalized = self.scaler.inverse_transform(convertToBCWH(self.args, normalized_data)).unstack()
            
            # 2. Add back trend
            relative_days = self._get_relative_days(denormalized)
            da_stacked = denormalized.stack(feature=['varOI','lat','lon']).transpose('N','feature',...)
            trend = self.detrend_model.predict(relative_days)
            
            detrended = xr.zeros_like(da_stacked)
            for pdf in da_stacked.pdf.values:
                for time in da_stacked.time.values:
                    if 'quantile' in da_stacked.dims:
                        for quantile in da_stacked['quantile'].values:
                        
                            detrended.loc[{'pdf':pdf,'time':time,'quantile':quantile}] = (da_stacked.sel(pdf=pdf, time=time, quantile=quantile) + trend)

                    else:
                        detrended.loc[{'pdf':pdf,'time':time}] = (da_stacked.sel(pdf=pdf, time=time) + trend)
            
            retrended = detrended.unstack()
            
            # 3. Add back seasonality
            final = retrended + self.seasonal_cycle_test[save_key]

        else:
            # Simple inverse normalization
            final = self.scaler.inverse_transform(convertToBCWH(self.args, normalized_data)).unstack()
        return convertToBCWH(self.args, final)



# Example usage
if __name__ == "__main__":
    import pickle as pkl
    
    # Load data
    tnvl_data = pkl.load(open('P2Final/tnvl_data','rb')).unstack()
    test_data = pkl.load(open('P2Final/test_data','rb')).unstack()
    tnvl_clim = pkl.load(open('P2Final/tnvl_clim','rb')).unstack()
    test_clim = pkl.load(open('P2Final/test_clim','rb')).unstack()
    
    # Create processors
    processor_train = DataProcessor()
    processor_test = DataProcessor()
    
    # Process data
    tnvl_clim = tnvl_clim.sel(N=tnvl_data.N)
    test_clim = test_clim.sel(N=test_data.N)
    train_processed = processor_train.fit_transform(tnvl_data, tnvl_clim, tnvl_data, tnvl_clim)
    test_processed = processor_test.fit_transform(tnvl_data, tnvl_clim, test_data, test_clim)
    
    # Restore data
    train_reversed = processor_train.inverse(train_processed)
    test_reversed = processor_test.inverse(test_processed)
    
# import numpy as np
# import xarray as xr
# from sklearn.linear_model import LinearRegression
# from datetime import datetime
# import pandas as pd
# from downscaling.data.GetData import convertToBCWH

# class MyNormalizer():
#     def __init__(self, standardize, model_name):
#         super().__init__()
#         self.standardize = standardize
#         self.model_name = model_name

#     def fit(self, refer):
#         if len(refer.dims)!=4:
#             de = 0
#         refer = refer.data
#         # # 假设X的形状是(sample, channels, width, height)
#         # 修改标准化策略，同时考虑时间和空间维度
#         if self.standardize == 'Norm0Spatial':
#             # 计算每个空间位置的均值和标准差
#             self.spatial_mean = refer.mean(axis=0, keepdims=True)  # 保持空间分布特征
#             self.spatial_std = refer.std(axis=0, keepdims=True)
            
#             # 计算全局均值和标准差用于整体归一化
#             self.global_mean = refer.mean()
#             self.global_std = refer.std()
#         elif self.standardize == 'Norm0SpatialV2':
#             # 计算每个空间位置的均值和标准差
#             self.spatial_mean = refer.mean(axis=0, keepdims=True)  # 保持空间分布特征
#             self.spatial_std = refer.std(axis=0, keepdims=True)
            
#             refer_norm = (refer - self.spatial_mean)/self.spatial_std
#             # 计算全局均值和标准差用于整体归一化
#             self.global_mean = refer_norm.mean()
#             self.global_std = refer_norm.std()

#         elif self.standardize == 'NormSpatioTemporal':
#             # 计算每个空间位置的统计特征
#             self.spatial_mean = refer.mean(axis=0, keepdims=True)
#             self.spatial_std = refer.std(axis=0, keepdims=True)
            
#             # 计算时间维度上的统计特征
#             self.temporal_mean = refer.mean(axis=(0, 2,3), keepdims=True)
#             self.temporal_std = refer.std(axis=(0, 2,3), keepdims=True)
#         elif self.standardize in ['Norm023','Norm023ThenSeason','Norm023OnlyX','Norm023OnlyXThenSeason']:
#             self.mean = refer.mean(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
#             self.std = refer.std(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
#         elif self.standardize in ['Norm0','Norm0ThenSeason','Norm0OnlyX','Norm0OnlyXThenSeason']:
#             self.mean = refer.mean(axis=(0, )).reshape(1, *refer.shape[1:])
#             self.std = refer.std(axis=(0, )).reshape(1, *refer.shape[1:])
#         elif self.standardize in ['Norm023stdSpatial']:
#             self.mean = refer.mean(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
#             self.std = refer.std(axis=(0, )).reshape(1, *refer.shape[1:])
#         elif self.standardize in ['NormGridMeanGlobalStd']:
#             self.mean = refer.mean(axis=(0, )).reshape(1, *refer.shape[1:])
#             self.std = refer.std(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
#         else:
#             raise ValueError('Error')
        
#         # if len(self.mean.shape)!=4:
#         #     de = 0
#         return 

#     def transform(self, x):
#         if self.standardize in ['Norm0Spatial','Norm0SpatialV2']:
#             # 先进行空间标准化
#             x_spatial = (x - self.spatial_mean) / self.spatial_std
#             # 再进行全局标准化
#             x_normalized = (x_spatial - self.global_mean) / self.global_std
#             return x_normalized
#         elif self.standardize == 'NormSpatioTemporal':
#             # 首先进行空间归一化
#             x_spatial = (x - self.spatial_mean) / self.spatial_std
            
#             # 然后进行时间维度的归一化
#             x_temporal = (x_spatial - self.temporal_mean) / self.temporal_std
#             return x_temporal
#         else:
#             x = (x - self.mean) / self.std
#             return x
    
#     def inverse_transform(self, x):
#         if self.standardize in ['Norm0Spatial','Norm0SpatialV2']:
#             x = x * self.global_std + self.global_mean
#             x = x * self.spatial_std + self.spatial_mean
#         else:
#             x = x * self.std + self.mean

#         return x
    
# class Distribution():
#     def __init__(self, ):
#         super().__init__()
    
#     def transform(self, x):
#         return np.sign(x) * np.log(1 + np.abs(x))

#     def inverse_transform(self, y):
#         return np.sign(y) * (np.exp(np.abs(y)) - 1)

# class DataProcessor:
#     """数据处理和还原的类"""
#     def __init__(self,args=None,model_name=None):
#         self.base_date = datetime(1994, 1, 1)
#         self.scaler = MyNormalizer(standardize=args.standardize,model_name=model_name)
#         self.args = args
#         self.seasonal_cycle_test = {}
        
#     def _get_relative_days(self, da):
#         """计算相对天数"""
#         dates = pd.to_datetime(da.N.values)
#         relative_days = (dates - self.base_date).days.values.reshape(-1, 1)
#         return relative_days

#     def fit(self, train_data, train_clim):
#         train_data = train_data.unstack()
#         train_clim = train_clim.unstack().sel(N=train_data.N)
#         if self.args.processor in [1,11]:
#             """拟合并转换数据"""
#             train_data = train_data.unstack()
#             train_clim = train_clim.unstack().sel(N=train_data.N)
#             # 1. 保存季节性循环
#             seasonal_cycle_train = train_clim.mean('pdf')
            
#             # 2. 去除季节性
#             train_deseasoned = train_data - seasonal_cycle_train
            
#             # 3. 拟合趋势模型
#             relative_days = self._get_relative_days(train_deseasoned)
#             da_tra_flat = (train_deseasoned.mean('pdf')
#                         .isel(time=0)
#                         .stack(feature=['varOI','lat','lon'])
#                         .transpose('N','feature'))
            
#             self.detrend_model = LinearRegression()
#             self.detrend_model.fit(relative_days, da_tra_flat.values)
            
#             # 4. 去除趋势
#             train_detrended = self._remove_trend(train_deseasoned)
            
#             # 5. 拟合归一化器
#             self.scaler.fit(refer=convertToBCWH(self.args, train_detrended))
#         elif self.args.processor in [2,12]:
#             """拟合并转换数据"""
#             train_data = train_data.unstack()
#             train_clim = train_clim.unstack().sel(N=train_data.N)
#             # 1. 保存季节性循环
#             seasonal_cycle_train = train_clim.mean('pdf').mean(['lat','lon']).expand_dims(lat=train_data.lat.data,lon=train_data.lon.data)
            
#             # 2. 去除季节性
#             train_deseasoned = train_data - seasonal_cycle_train
            
#             # 3. 拟合趋势模型
#             relative_days = self._get_relative_days(train_deseasoned)
#             da_tra_flat = (train_deseasoned.mean('pdf').mean(['lat','lon']).expand_dims(lat=train_data.lat.data,lon=train_data.lon.data)
#                         .isel(time=0)
#                         .stack(feature=['varOI','lat','lon'])
#                         .transpose('N','feature'))
            
#             self.detrend_model = LinearRegression()
#             self.detrend_model.fit(relative_days, da_tra_flat.values)
            
#             # 4. 去除趋势
#             train_detrended = self._remove_trend(train_deseasoned)
            
#             # 5. 拟合归一化器
#             self.scaler.fit(refer=convertToBCWH(self.args, train_detrended))
#         else:
#             if self.args.fcProcessor in ['subRA','RA']:
#                 self.scaler.fit(refer=convertToBCWH(self.args, train_data.unstack().mean('pdf').expand_dims(pdf=[0])))
#             elif self.args.fcProcessor in ['subCL','CL']:
#                 self.scaler.fit(refer=convertToBCWH(self.args, train_clim.unstack().mean('pdf').expand_dims(pdf=[0])))
#             else:
#                 raise ValueError('')
#         return 

#     def transform(self, test_data, test_clim, save_key):
#         """拟合并转换数据"""
#         test_data = test_data.unstack()
#         test_clim = test_clim.unstack().sel(N=test_data.N)
#         if self.args.processor in [1,11]:
#             # 1. 保存季节性循环
#             self.seasonal_cycle_test[save_key] = test_clim.mean('pdf').mean(['lat','lon'])
            
#             # 2. 去除季节性
#             test_deseasoned = test_data - self.seasonal_cycle_test[save_key]
            
#             # 4. 去除趋势
#             test_detrended = self._remove_trend(test_deseasoned)

#             # 6. 归一化
#             test_normalized = self.scaler.transform(convertToBCWH(self.args, test_detrended))
#         elif self.args.processor in [2,12]:
#             # 1. 保存季节性循环
#             self.seasonal_cycle_test[save_key] = test_clim.mean('pdf').mean(['lat','lon']).expand_dims(lat=test_data.lat.data,lon=test_data.lon.data)
            
#             # 2. 去除季节性
#             test_deseasoned = test_data - self.seasonal_cycle_test[save_key]
            
#             # 4. 去除趋势
#             test_detrended = self._remove_trend(test_deseasoned)

#             # 6. 归一化
#             test_normalized = self.scaler.transform(convertToBCWH(self.args, test_detrended))
#         else:
#             test_normalized = self.scaler.transform(convertToBCWH(self.args, test_data))

#         if self.args.processor in [10,11,12]:
#             test_normalized = Distribution().transform(test_normalized)

#         return test_normalized
        
#     def _remove_trend(self, da):
#         """去除趋势"""
#         relative_days = self._get_relative_days(da)
#         da_stacked = da.stack(feature=['varOI','lat','lon']).transpose('N','feature',...)
#         trend = self.detrend_model.predict(relative_days)
        
#         detrended = xr.zeros_like(da_stacked)
#         for pdf in da_stacked.pdf.values:
#             for time in da_stacked.time.values:
#                 trend = self.detrend_model.predict(relative_days)
#                 detrended.loc[{'pdf':pdf,'time':time}] = (
#                     da_stacked.sel(pdf=pdf, time=time) - trend
#                 )
        
#         return detrended.unstack('feature')
    
#     def inverse(self, normalized_data, save_key):
#         """还原数据到原始状态"""
#         if self.args.processor in [10,11,12]:
#             normalized_data = Distribution().inverse_transform(normalized_data)
#         if self.args.processor in [1,2,11,12]:
#             # 1. 逆归一化
#             denormalized = self.scaler.inverse_transform(convertToBCWH(self.args, normalized_data)).unstack()
            
#             # 2. 加回趋势
#             relative_days = self._get_relative_days(denormalized)
#             da_stacked = denormalized.stack(feature=['varOI','lat','lon']).transpose('N','feature',...)
#             trend = self.detrend_model.predict(relative_days)
            
#             detrended = xr.zeros_like(da_stacked)
#             for pdf in da_stacked.pdf.values:
#                 for time in da_stacked.time.values:
#                     if 'quantile' in da_stacked.dims:
#                         for quantile in da_stacked['quantile'].values:
                        
#                             detrended.loc[{'pdf':pdf,'time':time,'quantile':quantile}] = (da_stacked.sel(pdf=pdf, time=time, quantile=quantile) + trend)

#                     else:
#                         detrended.loc[{'pdf':pdf,'time':time}] = (da_stacked.sel(pdf=pdf, time=time) + trend)
            
#             retrended = detrended.unstack()
            
#             # 3. 加回季节性
#             final = retrended + self.seasonal_cycle_test[save_key]

#         else:
#             final = self.scaler.inverse_transform(convertToBCWH(self.args, normalized_data)).unstack()
#         return convertToBCWH(self.args, final)



# # 使用示例
# if __name__ == "__main__":
#     import pickle as pkl
    
#     # 加载数据
#     tnvl_data = pkl.load(open('P2Final/tnvl_data','rb')).unstack()
#     test_data = pkl.load(open('P2Final/test_data','rb')).unstack()
#     tnvl_clim = pkl.load(open('P2Final/tnvl_clim','rb')).unstack()
#     test_clim = pkl.load(open('P2Final/test_clim','rb')).unstack()
    
#     # 创建处理器
#     processor_train = DataProcessor()
#     processor_test = DataProcessor()
    
#     # 处理数据
#     tnvl_clim = tnvl_clim.sel(N=tnvl_data.N)
#     test_clim = test_clim.sel(N=test_data.N)
#     train_processed = processor_train.fit_transform(tnvl_data, tnvl_clim, tnvl_data, tnvl_clim)
#     test_processed = processor_test.fit_transform(tnvl_data, tnvl_clim, test_data, test_clim)
    
#     # 还原数据
#     train_reversed = processor_train.inverse(train_processed)
#     test_reversed = processor_test.inverse(test_processed)