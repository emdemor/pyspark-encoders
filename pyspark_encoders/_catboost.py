from typing import Dict

import pyspark.sql.functions as sf
from pyspark import keyword_only
from pyspark.sql.dataframe import DataFrame
from pyspark.ml import Estimator
from pyspark.ml.pipeline import Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import (
    HasInputCol,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)

from ._session import get_dataframe_spark_session


class CatboostEncoder(
    Estimator,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):

    a = Param(
        Params._dummy(),
        "a",
        "Parameter a",
        typeConverter=TypeConverters.toFloat,
    )

    sigma = Param(
        Params._dummy(),
        "sigma",
        "Parameter sigma",
        typeConverter=TypeConverters.toFloat,
    )

    targetCol = Param(
        Params._dummy(),
        "targetCol",
        "The target column name",
        typeConverter=TypeConverters.toString,
    )

    strategy = Param(
        Params._dummy(),
        "strategy",
        "Strategy to aggregate the target data",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, targetCol=None, a=1, sigma=None):
        super().__init__()
        self._setDefault(targetCol=None, a=1, sigma=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, targetCol=None, a=1, sigma=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, new_inputCol):
        """
        Setter for `inputCol`
        """
        return self.setParams(inputCol=new_inputCol)

    def setOutputCol(self, new_outputCol):
        """
        Setter for `outputCol`
        """
        return self.setParams(outputCol=new_outputCol)

    def setTargetCol(self, new_targetCol):
        """
        Setter for `targetCol`
        """
        return self.setParams(targetCol=new_targetCol)

    def getTargetCol(self):
        """
        Getter for `targetCol`
        """
        return self.getOrDefault(self.targetCol)

    def setA(self, new_a):
        """
        Setter for `a`
        """
        return self.setParams(a=new_a)

    def getA(self):
        """
        Getter for `a`
        """
        return self.getOrDefault(self.a)

    def setSigma(self, new_sigma):
        """
        Setter for `sigma`
        """
        return self.setParams(sigma=new_sigma)

    def getSigma(self):
        """
        Getter for `sigma`
        """
        return self.getOrDefault(self.sigma)

    def _fit(self, dataset):
        """
        Fits the CatboostEncoder to the input dataset
        """
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        target_col = self.getTargetCol()
        a = self.getA()
        sigma = self.getSigma()



        df_estimates = df.select(input_col, target_col).groupby(input_col).agg(sf.sum(target_col), sf.count(target_col))

        print(df_estimates.limit(10).toPandas())

        estimates = {row[input_col]: 1 for row in df_estimates.collect()}

        return CatboostEncoderModel(
            inputCol=input_col,
            outputCol=output_col,
            targetCol=target_col,
            categoriesTargetEstimates=estimates,
        )


class CatboostEncoderModel(
    Model,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    targetCol = Param(
        Params._dummy(),
        "targetCol",
        "The target column name",
        typeConverter=TypeConverters.toString,
    )

    categoriesTargetEstimates = Param(
        Params._dummy(),
        "categoriesTargetEstimates",
        "Target estimation for each category class",
    )

    @keyword_only
    def __init__(
        self,
        inputCol: str = None,
        outputCol: str = None,
        targetCol: str = None,
        categoriesTargetEstimates: Dict[str, float] = None,
    ):
        super().__init__()
        self._setDefault(targetCol=None)
        self._setDefault(categoriesTargetEstimates=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        inputCol: str = None,
        outputCol: str = None,
        targetCol: str = None,
        categoriesTargetEstimates: Dict[str, float] = None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, new_inputCol: str) -> None:
        """
        Setter for `inputCol`
        """
        return self.setParams(inputCol=new_inputCol)

    def setOutputCol(self, new_outputCol: str) -> None:
        """
        Setter for `outputCol`
        """
        return self.setParams(outputCol=new_outputCol)

    def setTargetCol(self, new_targetCol: str) -> None:
        """
        Setter for `targetCol`
        """
        return self.setParams(targetCol=new_targetCol)

    def setCategoriesTargetEstimates(
        self, new_categoriesTargetEstimates: Dict[str, float]
    ) -> None:
        """
        Setter for `categoriesTargetEstimates`
        """

        if not isinstance(new_categoriesTargetEstimates, dict):
            raise TypeError("The parameter `categoriesTargetEstimates` must be a dict.")

        if len(new_categoriesTargetEstimates) < 1:
            raise ValueError(
                "The parameter `categoriesTargetEstimates` cannot be empty."
            )

        return self.setParams(categoriesTargetEstimates=new_categoriesTargetEstimates)

    def getTargetCol(self) -> str:
        """
        Getter for `targetCol`
        """
        return self.getOrDefault(self.targetCol)

    def getCategoriesTargetEstimates(self) -> Dict[str, float]:
        """
        Getter for `categoriesTargetEstimates`
        """
        return self.getOrDefault(self.categoriesTargetEstimates)

    def _transform(self, dataset) -> DataFrame:
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        categories_target_estimates = self.getCategoriesTargetEstimates()
        spark = get_dataframe_spark_session(dataset)
        df_estimates = spark.createDataFrame(
            categories_target_estimates.items(), [input_col, output_col]
        )
        return dataset.join(df_estimates, on=input_col, how="left")
