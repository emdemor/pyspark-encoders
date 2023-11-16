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


class TargetEncoder(
    Estimator,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):

    """
    A class for encoding categorical columns using a given strategy
    to aggregate the target values over each category.

    Example
    --------
    >>> values = [("A", 1), ("A", 2), ("B", 6), ("B", 4), ("A", 3), ("B", 5)]
    >>> df = spark.createDataFrame(values, ["catColumn", "targetColumn"])
    >>> df.show()
    +---------+------------+
    |catColumn|targetColumn|
    +---------+------------+
    |        B|           6|
    |        B|           4|
    |        B|           5|
    |        A|           1|
    |        A|           2|
    |        A|           3|
    +---------+------------+
    >>> encoder = TargetEncoder(inputCol="catColumn", outputCol="encCatColumn",
    ...                         targetCol="targetColumn", strategy="mean")
    >>> encoder_model = encoder.fit(df)
    >>> transf_df = encoder_model.transform(df)
    >>> transf_df.show()
    +---------+------------+------------+
    |catColumn|targetColumn|encCatColumn|
    +---------+------------+------------+
    |        B|           6|         5.0|
    |        B|           4|         5.0|
    |        B|           5|         5.0|
    |        A|           1|         2.0|
    |        A|           2|         2.0|
    |        A|           3|         2.0|
    +---------+------------+------------+
    ...
    >>> prediction_values = [("A",), ("B",), ("A",), ("B",)]
    >>> prediction_df = spark.createDataFrame(prediction_values, ["catColumn"])
    >>> prediction_df.show()
    +---------+
    |catColumn|
    +---------+
    |        B|
    |        B|
    |        A|
    |        A|
    +---------+
    ...
    >>> encoder_model.transform(prediction_df).show()
    +---------+------------+
    |catColumn|encCatColumn|
    +---------+------------+
    |        B|         5.0|
    |        B|         5.0|
    |        A|         2.0|
    |        A|         2.0|
    +---------+------------+
    ...
    """

    target_encoder_strategies = {
        "mean": sf.mean,
        "avg": sf.mean,
        "min": sf.min,
        "max": sf.max,
        # TODO: sf.median is available from version 3.4.0
        "median": lambda column: sf.expr(f"percentile_approx({column}, 0.5)"),
    }

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
    def __init__(self, inputCol=None, outputCol=None, targetCol=None, strategy="mean"):
        super().__init__()
        self._setDefault(targetCol=None, strategy="mean")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, targetCol=None, strategy="mean"):
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

    def setStrategy(self, new_strategy):
        """
        Setter for `strategy`
        """
        return self.setParams(strategy=new_strategy)

    def getStrategy(self):
        """
        Getter for `strategy`
        """
        return self.getOrDefault(self.strategy)

    def _fit(self, dataset):
        """
        Fits the TargetEncoder to the input dataset
        """
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        target_col = self.getTargetCol()
        strategy = self.getStrategy()
        df_estimates = dataset.groupBy(input_col).agg(
            self._aggregate_column(target_col, strategy).alias(output_col)
        )
        estimates = {row[input_col]: row[output_col] for row in df_estimates.collect()}
        return TargetEncoderModel(
            inputCol=input_col,
            outputCol=output_col,
            targetCol=target_col,
            categoriesTargetEstimates=estimates,
        )

    def _aggregate_column(self, column: str, strategy: str):
        """
        Aggregates a column according to a defined strategy
        """
        try:
            aggregation_function = self.target_encoder_strategies[strategy]
        except KeyError as e:
            raise ValueError(
                f"Strategy '{strategy}' not recognized. "
                f"The implemented strategies are {list(self.target_encoder_strategies.keys())}."
            ) from e
        return aggregation_function(column)


class TargetEncoderModel(
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
