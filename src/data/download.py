"""Выгрузка измерений против вируса вакцинии из открытых баз.

Владелец: A. Критический путь ночи.

Каждая функция возвращает DataFrame в формате src.data.schema: smiles, IC50,
CC50, SI плюс doi, source, source_id и метаданные анализа. Концентрации уже
приведены к µM.

Главный факт, установленный на живом API ChEMBL: цитотоксичность не висит на
вирусной цели (там всего 5 записей CC50), она лежит на клеточных линиях. Но в
тех же статьях, где измеряли активность против вакцинии, есть 4759 измерений
токсичности. Поэтому пары IC50/CC50 собираются **по документу**, а не по цели:
выгрузка только по вирусной цели даёт 5 строк вместо 1164.

Все запросы идут через requests и кэшируются requests-cache: ночью пайплайн
перезапускается десятки раз, и без кэша каждый прогон заново ждёт сеть.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data import schema
from src.data.standardize import molecular_weight

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/"
CACHE_PATH = Path("data/raw/.http_cache")
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

# Страницы по 1000 записей рвутся по таймауту на медленной сети, по 200 —
# стабильны. Это наблюдение, а не вкусовщина.
PAGE_SIZE = 200
TIMEOUT = 120

# Что считается нашей целью. Эти строки попадают в data/SOURCES.md.
TARGET_ORGANISM_QUERY = "vaccinia"
ACTIVITY_TYPES_INHIBITION = ("IC50", "EC50")
ACTIVITY_TYPES_TOXICITY = ("CC50", "TC50", "CD50", "GI50")

# Клеточные линии, встречающиеся в описаниях антивирусных анализов. В записи
# анализа поля cell_chembl_id и assay_cell_type чаще всего пустые, поэтому
# линию приходится доставать из текста описания.
CELL_LINE_PATTERNS = (
    ("Vero", r"\bvero\b"),
    ("HeLa", r"\bhela\b"),
    ("HEL", r"\bhel\b"),
    ("BHK-21", r"\bbhk[- ]?21\b|\bbhk\b"),
    ("MDCK", r"\bmdck\b"),
    ("MT-4", r"\bmt[- ]?4\b"),
    ("CV-1", r"\bcv[- ]?1\b"),
    ("RK-13", r"\brk[- ]?13\b"),
    ("HFF", r"\bhff\b"),
    ("BSC-1", r"\bbsc[- ]?1\b"),
    ("A549", r"\ba549\b"),
    ("L929", r"\bl929\b"),
    ("MRC-5", r"\bmrc[- ]?5\b"),
    ("WI-38", r"\bwi[- ]?38\b"),
    ("Hep-2", r"\bhep[- ]?2\b"),
    ("CRFK", r"\bcrfk\b"),
    ("E6SM", r"\be6sm\b"),
)

# В ChEMBL цель цитотоксичности иногда записана не как линия клеток.
NON_CELL_LINE_TARGETS = {"admet", "unchecked", "no relevant target", ""}

# Статьи с панельным скринингом содержат цитотоксичность по каждому вирусу
# отдельно. Нужна та строка, что измерена в рукаве с вакцинией.
VACCINIA_PATTERN = re.compile(r"vaccinia|\bvacv\b|\bvv\b", re.IGNORECASE)

# Комментарии ChEMBL о сомнительных значениях. Такие строки по умолчанию
# отбраковываются: база сама говорит, что измерению нельзя доверять.
INVALID_VALIDITY_COMMENTS = (
    "Potential author error",
    "Outside typical range",
    "Non standard unit for type",
    "Potential missing data",
    "Author confirmed error",
)


def install_cache(expire_after: int = 24 * 3600) -> None:
    """Включить кэш HTTP для всех последующих запросов."""
    import requests_cache

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    requests_cache.install_cache(str(CACHE_PATH), expire_after=expire_after)


def _session() -> requests.Session:
    """Сессия с повторами: EBI отвечает 500 и рвёт соединение под нагрузкой."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _paginate(path: str, key: str, session: requests.Session, limit: int | None = None) -> list[dict]:
    """Пройти все страницы ответа ChEMBL по page_meta.next."""
    records: list[dict] = []
    url = BASE_URL + path
    while url:
        response = session.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        records.extend(payload[key])
        if limit is not None and len(records) >= limit:
            return records[:limit]
        nxt = payload.get("page_meta", {}).get("next")
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
    return records


def _chunks(values: list[str], size: int = 50):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_targets(session: requests.Session) -> list[dict]:
    """Цели ChEMBL по организму. Список не хардкодим: он попадает в SOURCES.md."""
    return _paginate(
        f"target.json?organism__icontains={TARGET_ORGANISM_QUERY}&limit={PAGE_SIZE}",
        "targets",
        session,
    )


def fetch_antiviral(target_ids: list[str], session: requests.Session, limit: int | None = None) -> list[dict]:
    """Активность против вируса: IC50 и EC50 по вирусным целям.

    EC50 и IC50 в антивирусных клеточных тестах означают одно и то же — 50%
    подавления репликации, — поэтому берём оба типа, а сам тип пишем в колонку
    assay_type_measured.
    """
    types = ",".join(ACTIVITY_TYPES_INHIBITION)
    records: list[dict] = []
    for chunk in _chunks(target_ids, 20):
        records += _paginate(
            f"activity.json?target_chembl_id__in={','.join(chunk)}"
            f"&standard_type__in={types}&limit={PAGE_SIZE}",
            "activities",
            session,
            limit=limit,
        )
        if limit is not None and len(records) >= limit:
            return records[:limit]
    return records


def fetch_cytotoxicity(document_ids: list[str], session: requests.Session) -> list[dict]:
    """Цитотоксичность из тех же статей. Линия клеток приходит в target_pref_name."""
    types = ",".join(ACTIVITY_TYPES_TOXICITY)
    records: list[dict] = []
    for chunk in _chunks(document_ids, 25):
        records += _paginate(
            f"activity.json?document_chembl_id__in={','.join(chunk)}"
            f"&standard_type__in={types}&limit={PAGE_SIZE}",
            "activities",
            session,
        )
    return records


def fetch_assay_cell_lines(assay_ids: list[str], session: requests.Session) -> dict[str, str]:
    """Линия клеток для каждого анализа.

    Поля cell_chembl_id и assay_cell_type в ChEMBL почти всегда пустые для
    антивирусных анализов, поэтому запасной путь — разбор описания.
    """
    mapping: dict[str, str] = {}
    for chunk in _chunks(assay_ids, 25):
        for assay in _paginate(
            f"assay.json?assay_chembl_id__in={','.join(chunk)}&limit={PAGE_SIZE}",
            "assays",
            session,
        ):
            mapping[assay["assay_chembl_id"]] = (
                assay.get("assay_cell_type")
                or cell_line_from_text(assay.get("description"))
                or ""
            )
    return mapping


def fetch_dois(document_ids: list[str], session: requests.Session) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in _chunks(document_ids, 25):
        for document in _paginate(
            f"document.json?document_chembl_id__in={','.join(chunk)}&limit={PAGE_SIZE}",
            "documents",
            session,
        ):
            mapping[document["document_chembl_id"]] = document.get("doi") or ""
    return mapping


def cell_line_from_text(text: str | None) -> str:
    """Вытащить название клеточной линии из описания анализа."""
    if not text:
        return ""
    lowered = str(text).lower()
    for name, pattern in CELL_LINE_PATTERNS:
        if re.search(pattern, lowered):
            return name
    return ""


def _measurement(activity: dict, cell_lines: dict[str, str]) -> tuple[dict | None, str]:
    """Привести одну запись активности к µM.

    Возвращает (запись, причина отказа). Причина непустая — строка непригодна и
    уходит в журнал исключений: условие задачи требует указать, почему строка
    не попала в обучающую базу.
    """
    smiles = activity.get("canonical_smiles")
    if not smiles:
        return None, "нет canonical_smiles в записи ChEMBL"

    value, _ = schema.parse_measurement(activity.get("standard_value"))
    if value is None:
        return None, "нечисловое standard_value"
    if not value > 0:
        return None, f"не положительное значение: {value}"

    units = activity.get("standard_units")
    if not units:
        return None, "не указаны единицы измерения"

    try:
        weight = molecular_weight(smiles) if schema.is_mass_unit(units) else None
        micromolar = schema.to_micromolar(value, units, weight)
    except schema.UnitConversionError as error:
        return None, f"единицы: {error}"

    low, high = schema.CONCENTRATION_RANGE_UM
    if not low <= micromolar <= high:
        return None, f"вне диапазона {low}–{high} µM: {micromolar:g}"

    relation = activity.get("standard_relation") or ""
    if relation == "=":
        relation = ""

    description = activity.get("assay_description") or ""
    target_name = activity.get("target_pref_name") or ""
    is_toxicity = (activity.get("standard_type") or "") in ACTIVITY_TYPES_TOXICITY

    # Линию клеток ищем в записи анализа, затем в описании. Цель измерения
    # годится как запасной путь только для цитотоксичности: там целью часто и
    # является линия. У антивирусных строк цель — сам вирус, и подставлять её
    # как линию клеток нельзя.
    cell_line = (
        cell_lines.get(activity.get("assay_chembl_id") or "", "")
        or cell_line_from_text(description)
    )
    if not cell_line and is_toxicity and target_name.strip().lower() not in NON_CELL_LINE_TARGETS:
        cell_line = target_name

    return {
        "value_um": micromolar,
        "qualifier": relation,
        "cell_line": cell_line,
        "measured_type": activity.get("standard_type") or "",
        "validity": activity.get("data_validity_comment") or "",
        "smiles": smiles,
        "assay_description": description,
        "mentions_vaccinia": bool(VACCINIA_PATTERN.search(description)),
    }, ""


def _representative(records: list[dict]) -> dict:
    """Свести несколько измерений одной молекулы в одной статье в одно.

    Точные измерения имеют приоритет над цензурированными: если в статье есть
    хотя бы одно значение без знака, оно и описывает соединение. Медиана
    берётся в лог-пространстве — концентрации распределены по порядкам.
    """
    exact = [record for record in records if not record["qualifier"]]
    pool = exact or records
    logs = np.log10([record["value_um"] for record in pool])
    median = float(np.median(logs))
    # Представитель — запись, ближайшая к медиане, чтобы сохранить её метаданные.
    chosen = min(pool, key=lambda record: abs(math.log10(record["value_um"]) - median))
    return {
        **chosen,
        "value_um": float(10**median),
        "n_measurements": len(records),
        "n_censored": sum(1 for record in records if record["qualifier"]),
    }


def _pick_cytotoxicity(candidates: list[dict], antiviral_cell_lines: set[str]) -> tuple[dict, str]:
    """Выбрать CC50 в пару к активности.

    Порядок приоритетов выведен из структуры реальных статей. Работы вроде
    панельного скрининга де Клерка измеряют цитотоксичность отдельно для каждого
    вирусного рукава: "Cytotoxicity against Vaccinia virus infected HEL cells".
    Брать оттуда рукав гриппа или ВИЧ — это другой эксперимент, поэтому:

      1. рукав с вакцинией в той же линии клеток;
      2. любой рукав с вакцинией;
      3. та же линия клеток;
      4. что есть в статье.

    Какое правило сработало, пишется в колонку: сдаваемая база должна позволять
    отличить честную пару от собранной на безрыбье.
    """
    known = {line for line in antiviral_cell_lines if line}
    vaccinia = [record for record in candidates if record.get("mentions_vaccinia")]

    same_line_vaccinia = [record for record in vaccinia if record["cell_line"] in known]
    if same_line_vaccinia:
        return _representative(same_line_vaccinia), "vaccinia_same_cell_line"
    if vaccinia:
        return _representative(vaccinia), "vaccinia_other_cell_line"

    same_line = [record for record in candidates if known and record["cell_line"] in known]
    if same_line:
        return _representative(same_line), "same_cell_line"
    return _representative(candidates), "any_in_document"


def build_table(
    antiviral: list[dict],
    cytotoxic: list[dict],
    cell_lines: dict[str, str],
    dois: dict[str, str],
    drop_invalid: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собрать пары по ключу (молекула, документ) и посчитать SI.

    Возвращает (таблица пар, журнал исключённых измерений с причинами).
    """
    inhibition: dict[tuple[str, str], list[dict]] = defaultdict(list)
    toxicity: dict[tuple[str, str], list[dict]] = defaultdict(list)
    excluded: list[dict] = []

    def collect(activities: list[dict], bucket: dict[tuple[str, str], list[dict]], role: str) -> None:
        for activity in activities:
            record, reason = _measurement(activity, cell_lines)
            if record is None:
                pass
            elif drop_invalid and record["validity"] in INVALID_VALIDITY_COMMENTS:
                reason = f"ChEMBL помечает значение как недостоверное: {record['validity']}"
                record = None

            key = (
                activity.get("molecule_chembl_id") or "",
                activity.get("document_chembl_id") or "",
            )
            if record is None:
                excluded.append({
                    "source": "chembl",
                    "role": role,
                    "source_id": key[0],
                    "document_id": key[1],
                    "smiles": activity.get("canonical_smiles") or "",
                    "standard_type": activity.get("standard_type") or "",
                    "standard_value": activity.get("standard_value"),
                    "standard_units": activity.get("standard_units") or "",
                    "reason": reason,
                })
                continue
            if all(key):
                bucket[key].append(record)
            else:
                excluded.append({
                    "source": "chembl",
                    "role": role,
                    "source_id": key[0],
                    "document_id": key[1],
                    "smiles": record["smiles"],
                    "standard_type": record["measured_type"],
                    "standard_value": record["value_um"],
                    "standard_units": "uM",
                    "reason": "нет идентификатора молекулы или статьи",
                })

    collect(antiviral, inhibition, "ингибирование")
    collect(cytotoxic, toxicity, "цитотоксичность")

    # Активность без цитотоксичности в той же статье — SI посчитать нельзя.
    for key in sorted(set(inhibition) - set(toxicity)):
        excluded.append({
            "source": "chembl",
            "role": "ингибирование",
            "source_id": key[0],
            "document_id": key[1],
            "smiles": inhibition[key][0]["smiles"],
            "standard_type": inhibition[key][0]["measured_type"],
            "standard_value": inhibition[key][0]["value_um"],
            "standard_units": "uM",
            "reason": "нет парной цитотоксичности в той же статье",
        })

    rows: list[dict] = []
    for key in sorted(set(inhibition) & set(toxicity)):
        molecule_id, document_id = key
        ic50 = _representative(inhibition[key])
        cc50, match = _pick_cytotoxicity(
            toxicity[key], {record["cell_line"] for record in inhibition[key]}
        )

        rows.append({
            "smiles": ic50["smiles"],
            "IC50": ic50["value_um"],
            "CC50": cc50["value_um"],
            "SI": schema.selectivity_index(ic50["value_um"], cc50["value_um"]),
            "doi": dois.get(document_id, ""),
            "source": "chembl",
            "source_id": molecule_id,
            "document_id": document_id,
            "virus": "vaccinia virus",
            "strain": "",
            "assay": ic50["assay_description"],
            "assay_type": ic50["measured_type"],
            "IC50_qualifier": ic50["qualifier"],
            "CC50_qualifier": cc50["qualifier"],
            "cell_line_ic50": ic50["cell_line"],
            "cell_line_cc50": cc50["cell_line"],
            "cc50_cell_line_match": match,
            "n_ic50": ic50["n_measurements"],
            "n_cc50": cc50["n_measurements"],
            # Оба значения цензурированы и равны: соединение неактивно и
            # тестировалось до одной и той же максимальной концентрации, а
            # SI = 1 здесь артефакт протокола, не измерение.
            "si_artifact": bool(
                ic50["qualifier"]
                and cc50["qualifier"]
                and math.isclose(ic50["value_um"], cc50["value_um"], rel_tol=1e-9)
            ),
        })

    return pd.DataFrame(rows), pd.DataFrame(excluded)


def from_chembl(config: dict | None = None) -> pd.DataFrame:
    """ChEMBL: активность против вакцинии в паре с цитотоксичностью из той же статьи."""
    settings = (config or {}).get("chembl", {})
    limit = settings.get("limit")
    drop_invalid = settings.get("drop_invalid", True)

    session = _session()

    targets = fetch_targets(session)
    target_ids = [target["target_chembl_id"] for target in targets]
    print(f"[chembl] целей по организму '{TARGET_ORGANISM_QUERY}': {len(target_ids)}")

    antiviral = fetch_antiviral(target_ids, session, limit=limit)
    print(f"[chembl] антивирусных активностей: {len(antiviral)}")

    document_ids = sorted({a["document_chembl_id"] for a in antiviral if a.get("document_chembl_id")})
    assay_ids = sorted({a["assay_chembl_id"] for a in antiviral if a.get("assay_chembl_id")})
    print(f"[chembl] статей: {len(document_ids)}, анализов: {len(assay_ids)}")

    cell_lines = fetch_assay_cell_lines(assay_ids, session)
    cytotoxic = fetch_cytotoxicity(document_ids, session)
    print(f"[chembl] измерений цитотоксичности в тех же статьях: {len(cytotoxic)}")

    dois = fetch_dois(document_ids, session)
    table, excluded = build_table(
        antiviral, cytotoxic, cell_lines, dois, drop_invalid=drop_invalid
    )
    print(f"[chembl] пар с IC50 и CC50: {len(table)}, "
          f"уникальных молекул: {table['source_id'].nunique() if len(table) else 0}")
    print(f"[chembl] исключено измерений: {len(excluded)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RAW_DIR / "chembl.csv", index=False)
    excluded.to_csv(INTERIM_DIR / "chembl_rejected.csv", index=False)
    return table


def from_pubchem(config: dict | None = None) -> pd.DataFrame:
    """PubChem BioAssay: поиск по вирусу вакцинии, AID → активности.

    Владелец A, после ChEMBL. Единицы в PubChem встречаются массовые, поэтому
    обязателен путь через schema.to_micromolar с молекулярной массой.
    """
    raise NotImplementedError("A: выгрузка PubChem")


def from_bindingdb(config: dict | None = None) -> pd.DataFrame:
    """BindingDB: выгрузка tsv и фильтр по цели. Берём, если останется время."""
    raise NotImplementedError("A: выгрузка BindingDB")


def from_starter_csv(config: dict | None = None) -> pd.DataFrame:
    """Стартовый датасет организаторов. Приводим к тому же формату."""
    path = (config or {}).get("starter_csv", "data/raw/starter.csv")
    frame = pd.read_csv(path)
    frame["source"] = "starter"
    return frame


def from_kaggle_train(config: dict | None = None) -> pd.DataFrame:
    """train.csv соревнования — обучающие данные в распределении теста.

    Это самый важный источник для метрики: ChEMBL даёт объём, но у него другое
    распределение SI, другие клеточные линии и другие протоколы. 320 строк
    организаторов — ровно то, на чём тест и построен.

    Запрет организаторов касается тестового файла: в обучающем измеренные CC50
    и IC50 использовать можно.

    У 81 строки IC50 не определён, и SI там задан меткой 10000 — это не
    измерение, а разметка «неактивно». Такие строки помечаются
    target_is_label, чтобы валидация не требовала от них согласованности
    SI = CC50 / IC50, а обучение могло их исключить переключателем.
    """
    from src.data import kaggle_data

    settings = (config or {}).get("kaggle_train", {})
    frame = kaggle_data.training_frame()

    ic50 = pd.to_numeric(frame.get("IC50"), errors="coerce")
    frame["target_is_label"] = ic50.isna()
    print(f"[kaggle_train] {len(frame)} строк, из них строк-меток SI=10000: "
          f"{int(frame['target_is_label'].sum())}")

    if settings.get("sentinels", "keep") == "drop":
        frame = frame[~frame["target_is_label"]].reset_index(drop=True)
        print(f"[kaggle_train] строки-метки исключены, осталось {len(frame)}")
    return frame


SOURCES = {
    "starter": from_starter_csv,
    "kaggle_train": from_kaggle_train,
    "chembl": from_chembl,
    "pubchem": from_pubchem,
    "bindingdb": from_bindingdb,
}
