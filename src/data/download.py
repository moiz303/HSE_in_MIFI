"""Выгрузка измерений против вируса вакцинии из открытых баз.

Владелец: A (стадия 1). Критический путь ночи — начинать с ChEMBL.

Каждая функция обязана возвращать DataFrame в формате src.data.schema:
smiles, IC50, CC50, SI плюс doi, source, source_id, virus, strain, cell_line.
Концентрации уже приведены к µM через schema.to_micromolar.

Все внешние запросы идут через кэш requests-cache: ночью пайплайн
перезапускается десятки раз, и без кэша каждый прогон заново ждёт сеть.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_PATH = Path("data/raw/.http_cache")

# Что именно считается нашей целью. Записываем рядом с данными, иначе через
# три часа никто не вспомнит, по какому фильтру качали.
TARGET_ORGANISM = "Vaccinia virus"
TARGET_KEYWORDS = ("vaccinia", "vacv", " vv ")
ACTIVITY_TYPES_INHIBITION = ("IC50", "EC50")
ACTIVITY_TYPES_TOXICITY = ("CC50", "TC50", "CD50")


def install_cache(expire_after: int = 24 * 3600) -> None:
    """Включить кэш HTTP для всех последующих запросов."""
    import requests_cache

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    requests_cache.install_cache(str(CACHE_PATH), expire_after=expire_after)


def from_chembl() -> pd.DataFrame:
    """ChEMBL через chembl_webresource_client.

    План запроса (A, заполнить):
      1. new_client.target.filter(organism__icontains="Vaccinia") → target_chembl_id
      2. new_client.activity.filter(target_chembl_id__in=..., standard_type__in=...)
         забрать standard_value, standard_units, standard_relation,
         molecule_chembl_id, document_chembl_id, assay_chembl_id, assay_description
      3. new_client.molecule по molecule_chembl_id → canonical_smiles
      4. new_client.document по document_chembl_id → doi
      5. свести IC50 и CC50 по паре (молекула, работа), посчитать SI

    Пара IC50/CC50 должна приходить из одной работы и, по возможности, из
    одной клеточной линии: SI из двух разных статей — это уже не индекс
    селективности, а отношение двух несвязанных чисел.
    """
    raise NotImplementedError("A: выгрузка ChEMBL")


def from_pubchem() -> pd.DataFrame:
    """PubChem BioAssay: поиск по вирусу вакцинии, AID → активности.

    Владелец A, после ChEMBL. Единицы в PubChem встречаются массовые,
    поэтому обязателен путь через schema.to_micromolar с молекулярной массой.
    """
    raise NotImplementedError("A: выгрузка PubChem")


def from_bindingdb() -> pd.DataFrame:
    """BindingDB: выгрузка tsv и фильтр по цели.

    Берём, если останется время после ChEMBL и PubChem.
    """
    raise NotImplementedError("A: выгрузка BindingDB")


def from_starter_csv(path: str | Path) -> pd.DataFrame:
    """Стартовый датасет организаторов. Приводим к тому же формату."""
    frame = pd.read_csv(path)
    frame["source"] = "starter"
    return frame


SOURCES = {
    "starter": from_starter_csv,
    "chembl": from_chembl,
    "pubchem": from_pubchem,
    "bindingdb": from_bindingdb,
}
