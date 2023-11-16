from pyspark.sql.dataframe import DataFrame


def get_dataframe_spark_session(dataframe: DataFrame):
    try:
        return dataframe.sparkSession
    except AttributeError as exception:
        if "no attribute 'sparkSession'" in str(exception):
            return dataframe.sql_ctx.sparkSession
        raise exception
