"""Custom kedro context to init pyspark"""

from pathlib import Path
from typing import Any, Dict, Union

from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from pluggy import PluginManager
from pyspark import SparkConf
from pyspark.sql import SparkSession


class RaftelContext(KedroContext):
    """Custom context used to initialize teh spark session"""

    def __init__(
            self,
            package_name: str,
            project_path: Union[Path, str],
            config_loader: ConfigLoader,
            hook_manager: PluginManager,
            env: str = None,
            extra_params: Dict[str, Any] = None,
    ):
        """Calls the super class constructor and the init_spark function"""
        super().__init__(
            package_name,
            project_path,
            config_loader,
            hook_manager,
            env,
            extra_params
            )
        self._spark_session = None
        self._init_spark()

    def _init_spark(self):
        """Initializes the spark context using the spark.yml config"""
        self._spark_session = SparkSession.getActiveSession()
        if self._spark_session is None:
            parameters = self._config_loader["spark"]
            spark_conf = SparkConf().setAll(
                    [(key, val) for key, val in parameters.items()]  # noqa: E501, pylint: disable=unnecessary-comprehension
                )

            spark_session_conf = (
                SparkSession.builder.appName(self._package_name)
                .master("local[*]")
                .config(conf=spark_conf)
            )
            self._spark_session = spark_session_conf.getOrCreate()
            self._spark_session.sparkContext.setLogLevel("WARN")
            print(self._spark_session.sparkContext.uiWebUrl)
