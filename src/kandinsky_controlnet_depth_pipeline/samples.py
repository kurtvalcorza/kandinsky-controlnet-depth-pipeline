"""Captioned-image dataset contract for adapting the generator: the pinned iNaturalist bird sample, seeded
splitting, generation prompts, BYOD loaders and CSV export.

The default dataset is **real** and narrow on purpose: 60 CC0-licensed, research-grade iNaturalist photographs of
six common North American birds (10 per species, one per observer per species), a subset of the corpus the
fleet's DINOv2 and ViT rows pinned on 2026-09-19, pinned here by photo id, byte size and SHA-256 of the served
`medium` JPEG (about 500 px on the longer side). Every file is fetched from the iNaturalist open-data bucket at
run time and refused on any byte-size or SHA-256 mismatch; the repository redistributes none of the photographs.
Each record keeps the observation id and observer login so every image is traceable to its public observation
page. Captions are generated from the species names by one template, so the adaptation teaches the generator
what these six names look like in this kind of photograph.

A record is ``{id, image, caption}``: a PIL image (or a path to one) and the caption used to generate it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MIN_TRAIN_RECORDS, MODEL_ID, image_digest, validate_dataset

CORPUS_NAME = "iNaturalist CC0 bird photographs (six species, 10 each)"
CORPUS_RELEASE = "iNaturalist open-data bucket, research-grade CC0 photos selected 2026-09-19 (fleet DINOv2/ViT corpus subset)"
CORPUS_BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/"
CORPUS_LICENSE = "CC0 1.0 (each photo's own license_code on iNaturalist; observers credited in the records)"
CORPUS_BYTES = 5_789_324
DEFAULT_CACHE_DIR = Path("weights") / "inat-birds"
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 6, "validation": 2, "test": 2}  # per species; 6 species -> 36 / 12 / 12
CAPTION_TEMPLATE = "a photo of a {common_name} ({scientific_name}), a wild bird photographed outdoors"
SPECIES: dict[str, tuple[str, str]] = {
    "song_sparrow": ("Melospiza melodia", "Song Sparrow"),
    "chipping_sparrow": ("Spizella passerina", "Chipping Sparrow"),
    "white_throated_sparrow": ("Zonotrichia albicollis", "White-throated Sparrow"),
    "dark_eyed_junco": ("Junco hyemalis", "Dark-eyed Junco"),
    "house_finch": ("Haemorhous mexicanus", "House Finch"),
    "american_goldfinch": ("Spinus tristis", "American Goldfinch"),
}
# (id, label, iNat photo id, iNat observation id, observer login, bytes, sha256 of the served
#  <photo id>/medium.<ext>, ext) — the bucket serves each photo under its original extension
#  (jpg or jpeg); the digest pins the served bytes
SAMPLE_RECORDS: tuple[tuple[str, str, int, int, str, int, str, str], ...] = (
    (
        "song_sparrow-00",
        "song_sparrow",
        129376982,
        79016324,
        "andywilson",
        43427,
        "7a9d9304a82f202e992655ec5f65477cd3d7c1dce03aa89a214c2daa38f9d61d",
        "jpg",
    ),
    (
        "song_sparrow-01",
        "song_sparrow",
        480991086,
        267636534,
        "lyneisfilm",
        162073,
        "11f77ff277dd2703c1000f2c787136ff0c3ca7ffad7017056892fe789d65efec",
        "jpg",
    ),
    (
        "song_sparrow-02",
        "song_sparrow",
        546060381,
        302980489,
        "swpollinators",
        27899,
        "4e70b9519c6e5f7384a4495b91b45465f2b1599f86491d8f9f9b12635a4046f6",
        "jpg",
    ),
    (
        "song_sparrow-03",
        "song_sparrow",
        308625896,
        177450028,
        "radrat",
        70961,
        "1211da4fdb24ae85ef0c6c3e2d03542c430457856aec661fe8f5f2de0027eee5",
        "jpeg",
    ),
    (
        "song_sparrow-04",
        "song_sparrow",
        494793016,
        275349085,
        "k-simpkins",
        58410,
        "255538cf450197257e86ed3d41dc69fb78e594434e9cb338c6314288c6cff26e",
        "jpg",
    ),
    (
        "song_sparrow-05",
        "song_sparrow",
        674054489,
        369029444,
        "ben142",
        220573,
        "1ae24622888d9d449ffd6b5c65cac1b9b14870fd8e12dbed8aa0acc2f5030123",
        "jpg",
    ),
    (
        "song_sparrow-06",
        "song_sparrow",
        339623726,
        193339933,
        "rawcomposition",
        25012,
        "d2cde085277a71886a2bf211a1eec26941a73752375708726a8af29aa4995a8b",
        "jpg",
    ),
    (
        "song_sparrow-07",
        "song_sparrow",
        222768957,
        130949329,
        "davidfbird",
        110773,
        "0feee62753f409d0aa365e9aa017ddef436ad70a2673dc847feb3b5386af27ff",
        "jpg",
    ),
    (
        "song_sparrow-08",
        "song_sparrow",
        181658744,
        107953669,
        "gcart043",
        98482,
        "a662a6abb24f42b256fb6e2d6f02c3e128a1053ef534f454461aa5cadcc03fe4",
        "jpeg",
    ),
    (
        "song_sparrow-09",
        "song_sparrow",
        148994242,
        90171417,
        "glennberry",
        102702,
        "64333977d24957d723a1d26886004f222d810557617685579f72a6b553e508fa",
        "jpg",
    ),
    (
        "chipping_sparrow-00",
        "chipping_sparrow",
        198992636,
        117809422,
        "k-simpkins",
        62563,
        "cf9f3b0c1863808e21af596b2e609b047ddbc28cb2ed076625e2546425ff0adf",
        "jpg",
    ),
    (
        "chipping_sparrow-01",
        "chipping_sparrow",
        248210057,
        144599194,
        "w_mark_c",
        193389,
        "497d0a0fef81c326bcc87b5d1eb97fe987d559b8f2b71bb60fe422ae34dce150",
        "jpg",
    ),
    (
        "chipping_sparrow-02",
        "chipping_sparrow",
        156350853,
        94266719,
        "ellyne",
        142332,
        "6a60ebac34476a372b4790a87d823432cdda8f72590930b5d18ae166cf4c7ba2",
        "jpeg",
    ),
    (
        "chipping_sparrow-03",
        "chipping_sparrow",
        16128796,
        11327134,
        "reuvenm",
        70532,
        "5ad36c9cdd6c92e225a1b8ab3c04d2f65bc4e971d0243958f8ac090b0996115c",
        "jpeg",
    ),
    (
        "chipping_sparrow-04",
        "chipping_sparrow",
        391300648,
        220982684,
        "carterdorscht",
        208567,
        "c974a676c3c227c2844ff822d429c784bc87bb41f40d5074a0ebadd4c6785b21",
        "jpeg",
    ),
    (
        "chipping_sparrow-05",
        "chipping_sparrow",
        339456083,
        193252042,
        "rawcomposition",
        46329,
        "41c9260ca9107430e3a8090ba01cebf3e3c25f2dad15bea4f77c8e824d9fc496",
        "jpg",
    ),
    (
        "chipping_sparrow-06",
        "chipping_sparrow",
        40457652,
        26076708,
        "andywilson",
        156894,
        "b70edd35e00fe672a39d5f0441ea4e9bb9fac19b0f2415ae13e22160bc6091a6",
        "jpeg",
    ),
    (
        "chipping_sparrow-07",
        "chipping_sparrow",
        84151914,
        52921135,
        "davidfbird",
        145714,
        "ea65ce5ed881ded9a8157b5756fa907948f41e0eefb6735b48a1c43993e977f9",
        "jpeg",
    ),
    (
        "chipping_sparrow-08",
        "chipping_sparrow",
        523674610,
        291074747,
        "rwp84",
        47983,
        "41eac0a5fb578b089f7524c4d2e6cb38c5508aa81815d6cfc46948c081daa800",
        "jpg",
    ),
    (
        "chipping_sparrow-09",
        "chipping_sparrow",
        292695018,
        168861389,
        "tim_kirsten",
        64812,
        "cbb2a03f5dbd2209d56f1cca8b49f342dfbaf44e51fe1014ead75b2018c6678d",
        "jpeg",
    ),
    (
        "white_throated_sparrow-00",
        "white_throated_sparrow",
        339621218,
        193338380,
        "rawcomposition",
        31152,
        "d1c08bfaca721bf0873437455b4cc010c6860d08b4777136c775007b0b9d07b6",
        "jpg",
    ),
    (
        "white_throated_sparrow-01",
        "white_throated_sparrow",
        166821399,
        99992799,
        "dziakj1",
        125954,
        "c595a41fbc8948b0d918b59117340dc320e2ba80d29e92bb9dacaaed5b404852",
        "jpeg",
    ),
    (
        "white_throated_sparrow-02",
        "white_throated_sparrow",
        469820434,
        261505977,
        "joy4birds",
        111767,
        "7ce091492c73c68395667bb45578dfff11457511b0958d1d0d320d1df3e55ceb",
        "jpg",
    ),
    (
        "white_throated_sparrow-03",
        "white_throated_sparrow",
        99351488,
        62040646,
        "bradenjudson",
        23399,
        "e103968e2a6c9efb6f0bcb548a6457aef950a4a720838a624b6796b90411538e",
        "jpeg",
    ),
    (
        "white_throated_sparrow-04",
        "white_throated_sparrow",
        177497964,
        105746665,
        "andywilson",
        29827,
        "65475d4842396f2488167453192d4aa834d2f1b40bcc78b420878c55ccf694d7",
        "jpeg",
    ),
    (
        "white_throated_sparrow-05",
        "white_throated_sparrow",
        628148203,
        344731686,
        "lavenderdame",
        106872,
        "36379abf3af51d865ba6e6804ba3dd48fc9efd12ecc0b7d97e03fc04a16178a8",
        "jpg",
    ),
    (
        "white_throated_sparrow-06",
        "white_throated_sparrow",
        104660609,
        65043951,
        "allan7",
        42443,
        "10791891945e83a0c908c14fea07a257a0f25f51c0e708bee35184213833a635",
        "jpeg",
    ),
    (
        "white_throated_sparrow-07",
        "white_throated_sparrow",
        250718938,
        145903421,
        "stevestevens",
        138735,
        "bd52e6d2f48c247f72db0fa393ec4fa13af8510c95f0ca9134cbef71caddb6d4",
        "jpeg",
    ),
    (
        "white_throated_sparrow-08",
        "white_throated_sparrow",
        15105971,
        10793852,
        "schylerbrown",
        31467,
        "6496e7e6d3abf13b1a538f769e3cc402280cf1e9a2a1ebdf81c68dcfd7ed01e2",
        "jpeg",
    ),
    (
        "white_throated_sparrow-09",
        "white_throated_sparrow",
        171460784,
        102554447,
        "w_mark_c",
        251287,
        "3828c41af9209b408fe0d8ec6541edf35d13fc730b74fd27a82980d4f3771137",
        "jpg",
    ),
    (
        "dark_eyed_junco-00",
        "dark_eyed_junco",
        172110799,
        102901486,
        "schylerbrown",
        182973,
        "185209c7a1111fc626a068e136ec3cfcdff3174d15f1c60af209d7b32fe7bef9",
        "jpeg",
    ),
    (
        "dark_eyed_junco-01",
        "dark_eyed_junco",
        46691943,
        29901256,
        "haida_gwaii",
        46823,
        "a92dca21e6e58c375fc313f0d1da2c86c11408beb0df8fd96fc60ae035ae80d5",
        "jpg",
    ),
    (
        "dark_eyed_junco-02",
        "dark_eyed_junco",
        707222551,
        386266764,
        "ben142",
        289608,
        "7bf320edf4d34a4848f3e9d175cfafa66cc5bab6a1a8f97c41c670263de3ff89",
        "jpg",
    ),
    (
        "dark_eyed_junco-03",
        "dark_eyed_junco",
        192557376,
        114006980,
        "k-simpkins",
        172398,
        "206f012add8a5d0fa434e07c51f1e7bd73a60c4d299a2202163fc662e1b03479",
        "jpeg",
    ),
    (
        "dark_eyed_junco-04",
        "dark_eyed_junco",
        8793471,
        6892999,
        "truthseqr",
        45302,
        "f8241eab39797c4e097a1432b13658466287d61ed515448457907c17e60cf5e2",
        "jpeg",
    ),
    (
        "dark_eyed_junco-05",
        "dark_eyed_junco",
        346777340,
        196961623,
        "zacharyfoster",
        46712,
        "ea8f9f0eebb4d2f86193c705344a8fab09bf634bc2217a0874ee57c4f0f5b4ab",
        "jpg",
    ),
    (
        "dark_eyed_junco-06",
        "dark_eyed_junco",
        274980085,
        159160633,
        "andy71",
        51209,
        "c54b45ca7fdc635bdb31eb89166c9fce84f0b0f8f0c17331fd5e42b582633c9b",
        "jpeg",
    ),
    (
        "dark_eyed_junco-07",
        "dark_eyed_junco",
        243303909,
        141959574,
        "andywilson",
        71321,
        "ee5308b6f93fb40a6d795a7d8ca6f2ef55b4844513f4268ef34827e1c5e4c4a6",
        "jpeg",
    ),
    (
        "dark_eyed_junco-08",
        "dark_eyed_junco",
        213798376,
        126031618,
        "nathanael15",
        57646,
        "d6b9e7d2dcc63cdd88b47cce328149aa9e8eb486ee2a5fc0f08b90601f2c7d9b",
        "jpg",
    ),
    (
        "dark_eyed_junco-09",
        "dark_eyed_junco",
        63482066,
        39977347,
        "chrisleearm",
        41870,
        "90573e814dbde965c70a934d9702110c40670aa22cc9f80ff8849db877d1fd72",
        "jpeg",
    ),
    (
        "house_finch-00",
        "house_finch",
        117990649,
        72375345,
        "kristen163",
        75945,
        "eeafad0dd2e91ecfe45c9d1f27dd0392a01bd81099549c60fd2e36a4b4342a9f",
        "jpeg",
    ),
    (
        "house_finch-01",
        "house_finch",
        389479656,
        220010434,
        "aster-asti",
        82128,
        "a8848197b4e7890e07538d492480c4275b75d04e10c1ae95aee91aaafe3319c5",
        "jpg",
    ),
    (
        "house_finch-02",
        "house_finch",
        176982307,
        105476125,
        "vicki936",
        22211,
        "c377fb361df0324c7a856d9344968886ece3b94bd67188c9325b8d2d284d3a2f",
        "jpeg",
    ),
    (
        "house_finch-03",
        "house_finch",
        697940852,
        381438133,
        "ben142",
        196735,
        "a89f8e0263fdabb404b462acaa592f5dd2ac88ee4615da444470de4a1fae82d5",
        "jpg",
    ),
    (
        "house_finch-04",
        "house_finch",
        72470599,
        45698380,
        "henrya",
        61724,
        "1b96d37a7078e1b725b80af4b10848da58b0d0c17a70c8ac01e326c0a749ee6b",
        "jpeg",
    ),
    (
        "house_finch-05",
        "house_finch",
        98576538,
        61594129,
        "enspring",
        46784,
        "8b355426d8fe6327f202c6a9458cee1b95de445bfbf7e48a4b5a2c7d0eb78a8c",
        "jpg",
    ),
    (
        "house_finch-06",
        "house_finch",
        80751781,
        50842166,
        "leahmfulton",
        44269,
        "6d6202de26f042d83ee6c5af550cd74e6ab10eb796eed2f78c83ac9e2368e4b7",
        "jpg",
    ),
    (
        "house_finch-07",
        "house_finch",
        630196420,
        345777550,
        "truthseqr",
        149455,
        "abb84d1e327dd82c07cbea3dd5583c07453e69b2cc220397b101e597da81bd6c",
        "jpg",
    ),
    (
        "house_finch-08",
        "house_finch",
        214612538,
        126483167,
        "hamiltonturner",
        124116,
        "6b7687640c4641b974865da04cf9eaf1f86b774ebc19678f2fc39e55c8648930",
        "jpeg",
    ),
    (
        "house_finch-09",
        "house_finch",
        213077180,
        125637342,
        "jnicat",
        25823,
        "e1e3baff8d0bd72339e3e49089f7f4f2c1dd383ff005e49a964a1bacc4f8ebb8",
        "jpeg",
    ),
    (
        "american_goldfinch-00",
        "american_goldfinch",
        84579952,
        53187208,
        "glennberry",
        59673,
        "72d36079e592e0a83c2f774f9073bfd4cc81253452c925d1673217ddd4b52a36",
        "jpeg",
    ),
    (
        "american_goldfinch-01",
        "american_goldfinch",
        12533322,
        9255418,
        "braincellsgone",
        55661,
        "6344e0125e74791f43ac6e07e5e1b9fbfce6d19bc62b6bb5d83b3caff9f7bcbc",
        "jpg",
    ),
    (
        "american_goldfinch-02",
        "american_goldfinch",
        131102823,
        80016788,
        "radrat",
        98990,
        "232a944f7e3351d4916a12ef2f6d598e7b007caaa95a9b064a14c1ba3af2a6ff",
        "jpeg",
    ),
    (
        "american_goldfinch-03",
        "american_goldfinch",
        175048222,
        104466897,
        "eug302",
        44231,
        "11c723482cc75fcc3a723ac1c0818a68e2fcf74c4ca684cf60195bf0d33f274e",
        "jpg",
    ),
    (
        "american_goldfinch-04",
        "american_goldfinch",
        68849595,
        43390778,
        "mefisher",
        154503,
        "66f07bc59bb3fdedd65a4537ebabd0cafd457826b8bf4bb633181f584a3edfd1",
        "jpg",
    ),
    (
        "american_goldfinch-05",
        "american_goldfinch",
        431916465,
        242278180,
        "k-simpkins",
        45413,
        "d76e7adf33e3a84ebec24dde5438965e62ebec8595755453973846340e4f460d",
        "jpg",
    ),
    (
        "american_goldfinch-06",
        "american_goldfinch",
        377136648,
        213398931,
        "nathan1177",
        66516,
        "7ad75838fdf2020a8e426e97507c7dd4355da93e6eece241128c28adbe302438",
        "jpg",
    ),
    (
        "american_goldfinch-07",
        "american_goldfinch",
        230801384,
        135330401,
        "enspring",
        42886,
        "78aebfb9b28c3e16dd9618a0e1ae06df67bfa4b8550c0879427d96915c475fed",
        "jpeg",
    ),
    (
        "american_goldfinch-08",
        "american_goldfinch",
        660472044,
        361884286,
        "ben142",
        270599,
        "733d64cd50c61334682f0862f5c7859ded34cae5776ee0bc6594fee400dc4876",
        "jpg",
    ),
    (
        "american_goldfinch-09",
        "american_goldfinch",
        294667312,
        169935316,
        "dande",
        163470,
        "39e7892e81eeef6af4887e61bc0998e17797688eb6394fcc8c9438391e875ee0",
        "jpeg",
    ),
)
SAMPLE_LABEL_SOURCE = f"{CORPUS_NAME}; {CORPUS_RELEASE}; {CORPUS_LICENSE}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def caption_for(label: str) -> str:
    """The template caption of a species key (the prompt the tutorial trains and generates with)."""
    if label not in SPECIES:
        raise ValueError(f"unknown species key {label!r}; expected one of {sorted(SPECIES)}")
    scientific, common = SPECIES[label]
    return CAPTION_TEMPLATE.format(common_name=common, scientific_name=scientific)


def photo_url(photo_id: int, ext: str = "jpg") -> str:
    """The served object for a pinned photo; `ext` is its recorded original extension (jpg or jpeg)."""
    if ext not in ("jpg", "jpeg"):
        raise ValueError(f"unsupported photo extension {ext!r}")
    return f"{CORPUS_BASE_URL}{photo_id}/medium.{ext}"


def observation_url(observation_id: int) -> str:
    return f"https://www.inaturalist.org/observations/{observation_id}"


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return every pinned photo (bytes keyed by record id) from the cache or the open-data bucket."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for rid, _label, photo_id, _obs, _user, size, digest, ext in SAMPLE_RECORDS:
        local = cache / f"{photo_id}.jpg"
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = photo_url(photo_id, ext)
            if fetcher is not None:
                data = fetcher(url)
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "dimer-kandinsky-controlnet-depth-tutorial/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (pinned https URL)
                    data = response.read()
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{rid} ({photo_id}/medium.{ext}): fetched {len(data)} bytes with sha256 "
                    f"{_sha256_bytes(data)[:16]}…, pinned {size} / {digest[:16]}…"
                )
            local.write_bytes(data)
        out[rid] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """Decode the verified photo bytes into `{id, image, caption, label}` records with their provenance."""
    from PIL import Image

    out = []
    for rid, label, photo_id, obs_id, user, _size, _digest, _ext in SAMPLE_RECORDS:
        if rid not in files:
            raise ValueError(f"corpus is missing {rid}")
        image = Image.open(io.BytesIO(files[rid]))
        image.load()
        out.append(
            {
                "id": rid,
                "image": image.convert("RGB"),
                "caption": caption_for(label),
                "label": label,
                "scientific_name": SPECIES[label][0],
                "common_name": SPECIES[label][1],
                "inat_photo_id": photo_id,
                "inat_observation_url": observation_url(obs_id),
                "observer": user,
            }
        )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded stratified draw per species: `sizes` counts per class for train / validation / test."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(str(record["label"]), []).append(dict(record))
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    for label in sorted(by_label):
        pool = by_label[label]
        rng.shuffle(pool)
        needed = sum(sizes.values())
        if len(pool) < needed:
            raise ValueError(f"{label}: only {len(pool)} records available, need {needed}")
        cursor = 0
        for name, per_class in sizes.items():
            out[name].extend(pool[cursor : cursor + per_class])
            cursor += per_class
    for name in out:
        rng.shuffle(out[name])
        out[name] = [{**r, "id": f"{name}-{i:03d}", "source_id": r["id"]} for i, r in enumerate(out[name])]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes)


def sample_prompts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The distinct captions of a split, in first-seen order (the prompts generation and scoring use)."""
    return list(dict.fromkeys(str(r["caption"]) for r in records))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image (by decoded-pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test, grouped by caption, after de-duplicating
    images. Every caption keeps at least one test record when it has three or more images."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    by_caption: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            by_caption.setdefault(record["caption"], []).append(record)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for caption in sorted(by_caption):
        pool = by_caption[caption]
        rng.shuffle(pool)
        n_test = max(1, round(len(pool) * test_fraction)) if len(pool) >= 3 else 0
        n_val = round(len(pool) * val_fraction) if len(pool) >= 3 else 0
        splits["test"].extend(pool[:n_test])
        splits["validation"].extend(pool[n_test : n_test + n_val])
        splits["train"].extend(pool[n_test + n_val :])
    for part in splits.values():
        rng.shuffle(part)
    if len(splits["train"]) < MIN_TRAIN_RECORDS:
        raise ValueError(f"split leaves {len(splits['train'])} training records; at least {MIN_TRAIN_RECORDS} are required")
    if not splits["test"]:
        raise ValueError("split leaves no test record; give at least one caption three or more images")
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, caption}` records from a directory or a zip holding `captions.csv` (columns `id`, `file`,
    `caption`) beside the image files; images are decoded, never extracted to disk."""
    from PIL import Image

    source = Path(path)
    if source.is_dir():
        table = (source / "captions.csv").read_text(encoding="utf-8")
        loader = lambda name: Image.open(source / name)  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "captions.csv" not in members:
            raise ValueError("BYOD zip must contain captions.csv")
        table = archive.read(members["captions.csv"]).decode("utf-8")
        loader = lambda name: Image.open(io.BytesIO(archive.read(members[name])))  # noqa: E731
    else:
        raise ValueError("BYOD datasets must be a directory or a .zip holding captions.csv and the image files")
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "file", "caption"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"captions.csv is missing columns {sorted(missing)}")
    out = []
    for row in rows:
        image = loader(row["file"])
        image.load()
        out.append({"id": row["id"], "image": image.convert("RGB"), "caption": row["caption"]})
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the captions table of a split (id, file, caption, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "file", "caption", "label", "observer", "inat_observation_url"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "file": f"{record['inat_photo_id']}.jpg" if record.get("inat_photo_id") else f"{record['id']}.jpg",
                    "caption": record["caption"],
                    "label": record.get("label", ""),
                    "observer": record.get("observer", ""),
                    "inat_observation_url": record.get("inat_observation_url", ""),
                }
            )
    return out


def dataset_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Validate every split and summarise the dataset (counts, captions, digests) for provenance exports."""
    summary: dict[str, Any] = {"model_id": MODEL_ID, "splits": {}}
    for name, records in splits.items():
        report = validate_dataset(records, min_records=1)
        summary["splits"][name] = {
            "n_records": report["n_records"],
            "n_captions": report["n_captions"],
            "shorter_side": report["shorter_side"],
            "centre_cropped": report["centre_cropped"],
            "digest": report["digest"],
        }
    summary["disjoint"] = check_split_disjoint(splits)
    digests = json.dumps({k: v["digest"] for k, v in summary["splits"].items()}, sort_keys=True)
    summary["digest"] = hashlib.sha256(digests.encode("utf-8")).hexdigest()
    return summary
