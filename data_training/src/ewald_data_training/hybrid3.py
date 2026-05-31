"""HybriD3/MatD3 structure ingestion helpers."""

from __future__ import annotations

import gzip
import json
import re
import shlex
import tarfile
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HYBRID3_BASE_URL = "https://materials.hybrid3.duke.edu"
STRUCTURE_EXTENSIONS = {
    ".cif",
    ".mcif",
    ".cell",
    ".ent",
    ".ins",
    ".pdb",
    ".in",
    ".poscar",
    ".res",
    ".vasp",
    ".xyz",
    ".zip",
    ".tgz",
    ".tar",
    ".geometry",
}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tar.gz", ".tgz"}
KNOWN_STRUCTURE_NAMES = {"poscar", "contcar", "geometry.in"}


@dataclass(slots=True)
class Hybrid3DatasetRecord:
    """One atomic-structure dataset discovered through HybriD3."""

    dataset_id: int
    caption: str
    compound_name: str = ""
    formula: str = ""
    group: str = ""
    dimensionality: str = ""
    primary_property: str = ""
    reference_title: str = ""
    reference_doi: str = ""
    sample_type: str = ""
    origin: str = ""
    visible: bool = True
    representative: bool = False
    space_group: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_payload(
        cls,
        payload: dict[str, Any],
    ) -> "Hybrid3DatasetRecord":
        system = payload.get("system") or {}
        primary_property = payload.get("primary_property") or {}
        reference = payload.get("reference") or {}
        origin = (
            "experimental"
            if payload.get("is_experimental")
            else "computational" if payload.get("computational") else ""
        )
        return cls(
            dataset_id=int(payload["pk"]),
            caption=str(payload.get("caption") or ""),
            compound_name=str(system.get("compound_name") or ""),
            formula=str(system.get("formula") or ""),
            group=str(system.get("group") or ""),
            dimensionality=str(system.get("dimensionality") or ""),
            primary_property=str(primary_property.get("name") or ""),
            reference_title=str(reference.get("title") or ""),
            reference_doi=str(reference.get("doi_isbn") or ""),
            sample_type=str(payload.get("sample_type") or ""),
            origin=origin,
            visible=bool(payload.get("visible", True)),
            representative=bool(payload.get("representative", False)),
            space_group=str(payload.get("space_group") or ""),
            metadata={
                "created": payload.get("created"),
                "created_by": payload.get("created_by"),
                "updated": payload.get("updated"),
                "updated_by": payload.get("updated_by"),
                "system": system,
                "reference": reference,
                "primary_unit": payload.get("primary_unit"),
                "secondary_property": payload.get("secondary_property"),
                "secondary_unit": payload.get("secondary_unit"),
                "extraction_method": payload.get("extraction_method"),
                "linked_to": payload.get("linked_to", []),
                "verified_by": payload.get("verified_by", []),
                "synthesis": payload.get("synthesis", []),
                "experimental": payload.get("experimental", []),
                "computational": payload.get("computational", []),
                "subsets": payload.get("subsets", []),
            },
        )

    @property
    def structure_id(self) -> str:
        return f"hybrid3_{self.dataset_id}"

    def as_catalog_entry(self, structure_path: str) -> dict[str, Any]:
        """Return an EWALD structure-catalog entry."""

        api_metadata = _api_metadata_summary(self)
        return {
            "structure_id": self.structure_id,
            "name": self.compound_name or self.caption or self.structure_id,
            "path": structure_path,
            "file_format": Path(structure_path).suffix.lstrip(".") or "poscar",
            "family": "hybrid_organic_inorganic",
            "phase_class": str(self.dimensionality),
            "source": f"HybriD3 dataset {self.dataset_id}",
            "license": "HybriD3 open database; verify redistribution terms",
            "tags": _compact_tags(
                [
                    self.primary_property,
                    self.sample_type,
                    self.origin,
                    self.group,
                    self.space_group,
                ]
            ),
            "metadata": api_metadata,
        }


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        for key, value in attrs:
            if key.lower() in {"href", "src"} and value:
                self.links.append(value)


def fetch_atomic_structure_datasets(
    *,
    base_url: str = HYBRID3_BASE_URL,
    page_size: int = 200,
    limit: int | None = None,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> list[Hybrid3DatasetRecord]:
    """Fetch visible HybriD3 atomic-structure dataset metadata."""

    client = session or retry_session()
    records: list[Hybrid3DatasetRecord] = []
    page = 1
    while True:
        response = client.get(
            urljoin(base_url, "/materials/datasets/"),
            params={"page": page, "page_size": page_size},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("results", []):
            record = Hybrid3DatasetRecord.from_api_payload(item)
            if not record.visible:
                continue
            if record.primary_property.lower() != "atomic structure":
                continue
            records.append(record)
            if limit and len(records) >= limit:
                return records
        if not payload.get("next"):
            break
        page += 1
    return records


def discover_dataset_file_links(
    dataset_id: int,
    *,
    base_url: str = HYBRID3_BASE_URL,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> list[str]:
    """Find downloadable structure-like files from a dataset HTML
    page."""

    client = session or retry_session()
    html_url = urljoin(base_url, f"/materials/dataset/{dataset_id}")
    response = client.get(html_url, timeout=timeout)
    response.raise_for_status()
    parser = _LinkParser()
    parser.feed(response.text)
    raw_links: list[str] = list(parser.links)
    jsmol_url = urljoin(base_url, f"/materials/get-jsmol-input/{dataset_id}")
    try:
        jsmol = client.get(jsmol_url, timeout=timeout)
        jsmol.raise_for_status()
        raw_links.extend(re.findall(r"/media/[^\s{}\"']+", jsmol.text))
    except requests.RequestException:
        pass
    links: list[str] = []
    for link in raw_links:
        absolute = urljoin(base_url, link)
        if "/media/data_files/" not in absolute:
            continue
        path = urlparse(absolute).path
        if is_structure_like_path(path):
            links.append(absolute)
    return sorted(set(links))


def download_structure_files(
    records: Iterable[Hybrid3DatasetRecord],
    *,
    output_root: str | Path,
    base_url: str = HYBRID3_BASE_URL,
    timeout: float = 20.0,
    session: requests.Session | None = None,
    fixture_root: str | Path | None = None,
) -> dict[str, Any]:
    """Download/convert structure files and write an EWALD catalog."""

    record_list = list(records)
    output_path = Path(output_root).expanduser().resolve()
    raw_dir = output_path / "raw"
    structures_dir = output_path / "structures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    client = session or retry_session()
    catalog_entries: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for record in record_list:
        links = _fixture_links(record.dataset_id, fixture_root)
        if not links:
            links = discover_dataset_file_links(
                record.dataset_id,
                base_url=base_url,
                timeout=timeout,
                session=client,
            )
        row = {
            "dataset_id": record.dataset_id,
            "structure_id": record.structure_id,
            "links": links,
            "status": "no_structure_file",
            "catalog_path": "",
        }
        for link in links:
            raw_file = _write_download(
                link,
                dataset_id=record.dataset_id,
                raw_dir=raw_dir,
                timeout=timeout,
                session=client,
                fixture_root=fixture_root,
            )
            converted = convert_structure_file(raw_file, structures_dir)
            if converted is None:
                continue
            rel_path = converted.relative_to(output_path)
            entry = record.as_catalog_entry(str(rel_path))
            entry["metadata"]["structure_file"] = (
                extract_structure_file_metadata(converted)
            )
            catalog_entries.append(entry)
            row["status"] = "ready"
            row["catalog_path"] = str(rel_path)
            break
        manifest_rows.append(row)

    catalog_path = output_path / "hybrid3_structure_catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {"structures": catalog_entries},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    manifest_path = output_path / "hybrid3_ingest_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in manifest_rows)
        + "\n",
        encoding="utf-8",
    )
    return {
        "catalog_path": catalog_path,
        "manifest_path": manifest_path,
        "records": len(record_list),
        "ready": len(catalog_entries),
        "missing": sum(1 for row in manifest_rows if row["status"] != "ready"),
    }


def convert_structure_file(
    raw_file: str | Path,
    output_dir: str | Path,
) -> Path | None:
    """Convert a downloaded structure file into a simulator-readable
    file."""

    raw_path = Path(raw_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if is_archive_path(raw_path):
        return _convert_archive(raw_path, output_path)
    if is_gzip_structure_path(raw_path):
        return _convert_gzip_structure(raw_path, output_path)

    text = _read_text(raw_path)
    inferred = infer_structure_format(raw_path, text)
    if inferred in {"cif", "mcif"}:
        target = output_path / _with_suffix(raw_path.name, f".{inferred}")
        target.write_bytes(raw_path.read_bytes())
        return target
    if inferred == "poscar":
        target = output_path / _with_suffix(raw_path.name, ".vasp")
        target.write_text(text, encoding="utf-8")
        return target
    converters = {
        "aims": _aims_geometry_to_poscar,
        "castep_cell": _castep_cell_to_poscar,
        "extended_xyz": _extended_xyz_to_poscar,
        "pdb": _pdb_to_poscar,
        "shelx": _shelx_to_poscar,
    }
    converter = converters.get(inferred)
    if converter is not None:
        poscar = converter(raw_path)
        if poscar:
            target = output_path / _with_suffix(raw_path.name, ".vasp")
            target.write_text(poscar, encoding="utf-8")
            return target
    return None


def is_structure_like_path(raw_path: str | Path) -> bool:
    """Return whether a URL/path name looks worth downloading."""

    parsed = urlparse(str(raw_path))
    path = parsed.path or str(raw_path)
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    full_suffix = _compound_suffix(Path(path))
    if name in KNOWN_STRUCTURE_NAMES:
        return True
    if name.endswith("geometry.in"):
        return True
    if suffix in STRUCTURE_EXTENSIONS:
        return True
    return full_suffix in ARCHIVE_SUFFIXES or is_gzip_structure_path(path)


def is_archive_path(raw_path: str | Path) -> bool:
    """Return whether a path is a supported archive container."""

    return _compound_suffix(Path(raw_path)) in ARCHIVE_SUFFIXES


def is_gzip_structure_path(raw_path: str | Path) -> bool:
    """Return whether a path is a gzip-wrapped supported structure
    file."""

    parsed = urlparse(str(raw_path))
    path = Path(parsed.path or str(raw_path))
    if _compound_suffix(path) == ".tar.gz":
        return False
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if not suffixes or suffixes[-1] != ".gz":
        return False
    inner_name = path.name[: -len(".gz")]
    return is_structure_like_path(inner_name)


def infer_structure_format(
    raw_path: str | Path, text: str | None = None
) -> str:
    """Infer a HybriD3 structure file variant from name and content."""

    path = Path(raw_path)
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return "cif"
    if suffix == ".mcif":
        return "mcif"
    if suffix in {".vasp", ".poscar"} or name in {"poscar", "contcar"}:
        return "poscar"
    content = text if text is not None else _read_text(path)
    lowered = content.lower()
    if _looks_like_cif(content):
        return "cif"
    if _looks_like_poscar(content):
        return "poscar"
    if "lattice_vector" in lowered and (
        "\natom " in lowered or "\natom_frac" in lowered
    ):
        return "aims"
    if "%block lattice_cart" in lowered or "%block positions_" in lowered:
        return "castep_cell"
    if _looks_like_extended_xyz(content):
        return "extended_xyz"
    if "cryst1" in lowered[:5000]:
        return "pdb"
    if _looks_like_shelx(content):
        return "shelx"
    return ""


def extract_structure_file_metadata(
    structure_file: str | Path,
) -> dict[str, Any]:
    """Extract lightweight crystallographic metadata from a structure
    file."""

    path = Path(structure_file)
    text = _read_text(path)
    inferred = infer_structure_format(path, text)
    metadata: dict[str, Any] = {
        "file_name": path.name,
        "format": inferred or path.suffix.lstrip(".").lower(),
    }
    if inferred in {"cif", "mcif"}:
        metadata.update(_extract_cif_metadata(text))
    elif inferred == "poscar":
        metadata.update(_extract_poscar_metadata(text))
    return _drop_empty(metadata)


def load_fixture_records(
    fixture_root: str | Path,
) -> list[Hybrid3DatasetRecord]:
    """Load offline test records from a fixture directory."""

    fixture_path = Path(fixture_root)
    payload = json.loads((fixture_path / "datasets_page1.json").read_text())
    return [
        Hybrid3DatasetRecord.from_api_payload(item)
        for item in payload.get("results", [])
        if (item.get("primary_property") or {}).get("name")
        == "atomic structure"
    ]


def retry_session(
    *,
    retries: int = 4,
    backoff_factor: float = 1.0,
) -> requests.Session:
    """Return a requests session with conservative read-timeout
    retries."""

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    client = requests.Session()
    client.mount("https://", adapter)
    client.mount("http://", adapter)
    client.headers.update({"User-Agent": "EWALD data-training/0"})
    return client


def _write_download(
    link: str,
    *,
    dataset_id: int,
    raw_dir: Path,
    timeout: float,
    session: requests.Session,
    fixture_root: str | Path | None,
) -> Path:
    parsed = urlparse(link)
    filename = _safe_filename(
        Path(parsed.path).name or f"dataset_{dataset_id}.dat"
    )
    target = raw_dir / f"dataset_{dataset_id}_{filename}"
    fixture_file = _fixture_file(link, fixture_root)
    if fixture_file is not None:
        target.write_bytes(fixture_file.read_bytes())
        return target
    response = session.get(link, timeout=timeout)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def _fixture_links(
    dataset_id: int,
    fixture_root: str | Path | None,
) -> list[str]:
    if fixture_root is None:
        return []
    fixture_path = Path(fixture_root)
    html_path = fixture_path / f"dataset_{dataset_id}.html"
    if not html_path.exists():
        return []
    parser = _LinkParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return [urljoin(HYBRID3_BASE_URL, link) for link in parser.links]


def _fixture_file(
    link: str,
    fixture_root: str | Path | None,
) -> Path | None:
    if fixture_root is None:
        return None
    fixture_path = Path(fixture_root)
    filename = Path(urlparse(link).path).name
    candidate = fixture_path / filename
    if candidate.exists():
        return candidate
    return None


def _convert_archive(raw_path: Path, output_path: Path) -> Path | None:
    member_dir = (
        output_path / "_archive_members" / _safe_filename(raw_path.stem)
    )
    member_dir.mkdir(parents=True, exist_ok=True)
    members: list[tuple[str, bytes]] = []
    if raw_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(raw_path) as archive:
            for name in archive.namelist():
                if name.endswith("/") or not is_structure_like_path(name):
                    continue
                members.append((name, archive.read(name)))
    elif tarfile.is_tarfile(raw_path):
        with tarfile.open(raw_path) as archive:
            for member in archive.getmembers():
                if not member.isfile() or not is_structure_like_path(
                    member.name
                ):
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    members.append((member.name, handle.read()))
    for name, payload in sorted(
        members, key=lambda item: _format_priority(item[0])
    ):
        member_path = member_dir / _safe_filename(Path(name).name)
        member_path.write_bytes(payload)
        converted = convert_structure_file(member_path, output_path)
        if converted is not None:
            return converted
    return None


def _convert_gzip_structure(raw_path: Path, output_path: Path) -> Path | None:
    member_dir = (
        output_path / "_archive_members" / _safe_filename(raw_path.stem)
    )
    member_dir.mkdir(parents=True, exist_ok=True)
    member_name = raw_path.name[: -len(".gz")]
    member_path = member_dir / _safe_filename(member_name)
    with gzip.open(raw_path, "rb") as handle:
        member_path.write_bytes(handle.read())
    return convert_structure_file(member_path, output_path)


def _format_priority(raw_path: str | Path) -> tuple[int, str]:
    inferred = infer_structure_format(raw_path, "")
    order = {
        "cif": 0,
        "mcif": 1,
        "poscar": 2,
        "aims": 3,
        "castep_cell": 4,
        "pdb": 5,
        "shelx": 6,
        "extended_xyz": 7,
    }
    return order.get(inferred, 99), str(raw_path)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _looks_like_cif(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("data_") or "_cell_length_a" in text[:5000]


def _looks_like_poscar(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    try:
        float(lines[1].split()[0])
        for index in (2, 3, 4):
            values = [float(item) for item in lines[index].split()[:3]]
            if len(values) != 3:
                return False
    except (IndexError, ValueError):
        return False
    return True


def _looks_like_extended_xyz(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    try:
        atom_count = int(lines[0].strip())
    except ValueError:
        return False
    return (
        atom_count > 0
        and len(lines) >= atom_count + 2
        and ("lattice=" in lines[1].lower())
    )


def _looks_like_shelx(text: str) -> bool:
    upper = text.upper()
    return "CELL " in upper and "SFAC " in upper


def _with_suffix(raw_name: str, suffix: str) -> str:
    name = _safe_filename(raw_name)
    path = Path(name)
    if path.suffix.lower() != suffix.lower():
        name = f"{path.stem}{suffix}"
    return name


def _compound_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] == [".tar", ".gz"]:
        return ".tar.gz"
    if suffixes:
        return suffixes[-1]
    return ""


def _lattice_from_cell(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> list[list[float]]:
    import math

    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * math.cos(gamma), b * math.sin(gamma), 0.0
    cx = c * math.cos(beta)
    cy = (
        c
        * (math.cos(alpha) - math.cos(beta) * math.cos(gamma))
        / max(math.sin(gamma), 1.0e-12)
    )
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]


def _poscar_text(
    *,
    comment: str,
    lattice: list[list[float]],
    species: list[str],
    coords: list[list[float]],
    coord_kind: str,
) -> str:
    ordered_species = sorted(dict.fromkeys(species))
    counts = [species.count(symbol) for symbol in ordered_species]
    coord_map: dict[str, list[list[float]]] = {
        symbol: [] for symbol in ordered_species
    }
    for symbol, coord in zip(species, coords):
        coord_map[symbol].append(coord)
    lines = [
        comment,
        "1.0",
        *(
            "  " + "  ".join(f"{value:.12f}" for value in row)
            for row in lattice
        ),
        "  ".join(ordered_species),
        "  ".join(str(count) for count in counts),
        coord_kind,
    ]
    for symbol in ordered_species:
        for coord in coord_map[symbol]:
            lines.append("  " + "  ".join(f"{value:.12f}" for value in coord))
    return "\n".join(lines) + "\n"


def _aims_geometry_to_poscar(path: Path) -> str | None:
    lattice: list[list[float]] = []
    species: list[str] = []
    coords_cart: list[list[float]] = []
    coords_frac: list[list[float]] = []
    for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if parts[0] == "lattice_vector" and len(parts) >= 4:
            lattice.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "atom" and len(parts) >= 5:
            coords_cart.append(
                [float(parts[1]), float(parts[2]), float(parts[3])]
            )
            species.append(parts[4])
        elif parts[0] == "atom_frac" and len(parts) >= 5:
            coords_frac.append(
                [float(parts[1]), float(parts[2]), float(parts[3])]
            )
            species.append(parts[4])
    if len(lattice) != 3 or not species:
        return None
    coord_kind = "Direct" if coords_frac else "Cartesian"
    coords = coords_frac or coords_cart
    return _poscar_text(
        comment=f"Converted from HybriD3 FHI-aims {path.name}",
        lattice=lattice,
        species=species,
        coords=coords,
        coord_kind=coord_kind,
    )


def _castep_cell_to_poscar(path: Path) -> str | None:
    blocks = _parse_percent_blocks(_read_text(path))
    lattice = _numeric_block(blocks.get("lattice_cart", []), expected=3)
    positions = blocks.get("positions_frac") or blocks.get("positions_abs")
    if len(lattice) != 3 or not positions:
        return None
    species: list[str] = []
    coords: list[list[float]] = []
    for line in positions:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
        species.append(parts[0])
    if not species:
        return None
    coord_kind = "Direct" if "positions_frac" in blocks else "Cartesian"
    return _poscar_text(
        comment=f"Converted from HybriD3 CASTEP cell {path.name}",
        lattice=lattice,
        species=species,
        coords=coords,
        coord_kind=coord_kind,
    )


def _parse_percent_blocks(text: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        lower = clean.lower()
        if lower.startswith("%block"):
            current = lower.split(maxsplit=1)[1].strip()
            blocks[current] = []
        elif lower.startswith("%endblock"):
            current = None
        elif current is not None:
            blocks[current].append(clean)
    return blocks


def _numeric_block(lines: list[str], *, expected: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in lines:
        try:
            values = [float(item) for item in line.split()[:expected]]
        except ValueError:
            continue
        if len(values) == expected:
            rows.append(values)
    return rows


def _extended_xyz_to_poscar(path: Path) -> str | None:
    text = _read_text(path)
    lines = text.splitlines()
    if not _looks_like_extended_xyz(text):
        return None
    atom_count = int(lines[0].strip())
    match = re.search(r'Lattice="([^"]+)"', lines[1])
    if not match:
        return None
    lattice_values = [float(item) for item in match.group(1).split()]
    if len(lattice_values) != 9:
        return None
    lattice = [
        lattice_values[0:3],
        lattice_values[3:6],
        lattice_values[6:9],
    ]
    species: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + atom_count]:
        parts = line.split()
        if len(parts) < 4:
            return None
        species.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return _poscar_text(
        comment=f"Converted from HybriD3 extended XYZ {path.name}",
        lattice=lattice,
        species=species,
        coords=coords,
        coord_kind="Cartesian",
    )


def _pdb_to_poscar(path: Path) -> str | None:
    lattice: list[list[float]] | None = None
    species: list[str] = []
    coords: list[list[float]] = []
    for line in _read_text(path).splitlines():
        if line.startswith("CRYST1"):
            try:
                parts = line.split()
                lattice = _lattice_from_cell(
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                )
            except (IndexError, ValueError):
                return None
        elif line.startswith(("ATOM", "HETATM")):
            try:
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError:
                continue
            element = line[76:78].strip()
            if not element:
                element = re.sub("[^A-Za-z]", "", line[12:16]).strip()
            species.append(element.capitalize())
    if lattice is None or not species:
        return None
    return _poscar_text(
        comment=f"Converted from HybriD3 PDB {path.name}",
        lattice=lattice,
        species=species,
        coords=coords,
        coord_kind="Cartesian",
    )


def _shelx_to_poscar(path: Path) -> str | None:
    cell: list[float] | None = None
    sfac: list[str] = []
    species: list[str] = []
    coords: list[list[float]] = []
    for line in _read_text(path).splitlines():
        parts = line.split()
        if not parts:
            continue
        keyword = parts[0].upper()
        if keyword == "CELL" and len(parts) >= 8:
            cell = [float(item) for item in parts[2:8]]
        elif keyword == "SFAC" and len(parts) >= 2:
            sfac = [item.capitalize() for item in parts[1:]]
        elif sfac and len(parts) >= 5:
            try:
                species_index = int(parts[1]) - 1
                coord = [float(parts[2]), float(parts[3]), float(parts[4])]
            except ValueError:
                continue
            if 0 <= species_index < len(sfac):
                species.append(sfac[species_index])
                coords.append(coord)
    if cell is None or not species:
        return None
    return _poscar_text(
        comment=f"Converted from HybriD3 SHELX {path.name}",
        lattice=_lattice_from_cell(*cell),
        species=species,
        coords=coords,
        coord_kind="Direct",
    )


def _api_metadata_summary(record: Hybrid3DatasetRecord) -> dict[str, Any]:
    system = record.metadata.get("system") or {}
    reference = record.metadata.get("reference") or {}
    components = {
        "compound_formula": record.formula,
        "organic_formula": _clean_text(system.get("organic")),
        "inorganic_formula": _clean_text(system.get("inorganic")),
        "group_formula": _clean_text(system.get("group")),
        "iupac_name": _clean_text(system.get("iupac")),
        "element_counts": {
            "total": parse_formula_counts(record.formula),
            "organic": parse_formula_counts(system.get("organic")),
            "inorganic": parse_formula_counts(system.get("inorganic")),
        },
    }
    return _drop_empty(
        {
            "dataset_id": record.dataset_id,
            "caption": record.caption,
            "formula": record.formula,
            "components": components,
            "system": _drop_empty(
                {
                    "id": system.get("id"),
                    "compound_name": record.compound_name,
                    "description": _clean_text(system.get("description")),
                    "message": _clean_text(system.get("message")),
                    "dimensionality": record.dimensionality,
                    "n": _clean_text(system.get("n")),
                    "last_update": system.get("last_update"),
                    "derived_to_from": system.get("derived_to_from"),
                    "tags": system.get("tags"),
                }
            ),
            "property": _drop_empty(
                {
                    "primary": record.primary_property,
                    "primary_unit": (
                        record.metadata.get("primary_unit") or {}
                    ).get("label"),
                    "secondary": (
                        record.metadata.get("secondary_property") or {}
                    ).get("name"),
                    "secondary_unit": (
                        record.metadata.get("secondary_unit") or {}
                    ).get("label"),
                }
            ),
            "provenance": _drop_empty(
                {
                    "origin": record.origin,
                    "sample_type": record.sample_type,
                    "extraction_method": record.metadata.get(
                        "extraction_method"
                    ),
                    "created": record.metadata.get("created"),
                    "created_by": record.metadata.get("created_by"),
                    "updated": record.metadata.get("updated"),
                    "updated_by": record.metadata.get("updated_by"),
                    "representative": record.representative,
                    "visible": record.visible,
                    "verified_by": record.metadata.get("verified_by"),
                }
            ),
            "reference": _drop_empty(
                {
                    "id": reference.get("id"),
                    "title": record.reference_title,
                    "journal": reference.get("journal"),
                    "volume": reference.get("vol"),
                    "pages_start": reference.get("pages_start"),
                    "pages_end": reference.get("pages_end"),
                    "year": reference.get("year"),
                    "doi": record.reference_doi,
                }
            ),
            "linked_to": record.metadata.get("linked_to", []),
            "computational": record.metadata.get("computational", []),
            "synthesis": record.metadata.get("synthesis", []),
            "experimental": record.metadata.get("experimental", []),
            "crystallography": _api_crystallography_summary(record),
        }
    )


def _api_crystallography_summary(
    record: Hybrid3DatasetRecord,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"space_group": record.space_group}
    subsets = record.metadata.get("subsets") or []
    if not subsets:
        return _drop_empty(summary)
    first = subsets[0] or {}
    values: list[float] = []
    raw_values: list[str] = []
    for datapoint in first.get("datapoints") or []:
        for value in datapoint.get("values") or []:
            formatted = str(value.get("formatted") or "")
            parsed = _first_float(formatted)
            if parsed is not None:
                values.append(parsed)
                raw_values.append(formatted)
                break
    if len(values) >= 6:
        summary["cell"] = {
            "a": values[0],
            "b": values[1],
            "c": values[2],
            "alpha": values[3],
            "beta": values[4],
            "gamma": values[5],
            "raw_values": raw_values[:6],
            "source": "hybrid3_subset_datapoint_order",
        }
    summary["crystal_system"] = _clean_text(first.get("crystal_system"))
    summary["subsets"] = subsets
    return _drop_empty(summary)


def _extract_cif_metadata(text: str) -> dict[str, Any]:
    scalars = _parse_cif_scalars(text)
    cell = _drop_empty(
        {
            "a": _first_float(scalars.get("_cell_length_a")),
            "b": _first_float(scalars.get("_cell_length_b")),
            "c": _first_float(scalars.get("_cell_length_c")),
            "alpha": _first_float(scalars.get("_cell_angle_alpha")),
            "beta": _first_float(scalars.get("_cell_angle_beta")),
            "gamma": _first_float(scalars.get("_cell_angle_gamma")),
            "volume": _first_float(scalars.get("_cell_volume")),
            "z": _first_float(scalars.get("_cell_formula_units_z")),
        }
    )
    formulas = _drop_empty(
        {
            "sum": _clean_cif_value(scalars.get("_chemical_formula_sum")),
            "moiety": _clean_cif_value(
                scalars.get("_chemical_formula_moiety")
            ),
            "oxdiff": _clean_cif_value(
                scalars.get("_chemical_oxdiff_formula")
            ),
        }
    )
    atom_sites = _parse_cif_atom_sites(text)
    return _drop_empty(
        {
            "cell": cell,
            "crystal_system": _clean_cif_value(
                scalars.get("_space_group_crystal_system")
                or scalars.get("_symmetry_cell_setting")
            ),
            "space_group_name": _clean_cif_value(
                scalars.get("_space_group_name_H-M_alt")
                or scalars.get("_symmetry_space_group_name_H-M")
            ),
            "formulas": formulas,
            "formula_element_counts": parse_formula_counts(
                formulas.get("sum")
            ),
            "chemical_name": _clean_cif_value(
                scalars.get("_chemical_name_systematic")
                or scalars.get("_chemical_name_common")
            ),
            "absolute_configuration": _clean_cif_value(
                scalars.get("_chemical_absolute_configuration")
            ),
            "density": _first_float(
                scalars.get("_exptl_crystal_density_diffrn")
            ),
            "r_factor": _first_float(
                scalars.get("_refine_ls_R_factor_gt")
                or scalars.get("_refine_ls_R_factor_all")
            ),
            "atom_sites": atom_sites,
        }
    )


def _extract_poscar_metadata(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return {}
    try:
        scale = float(lines[1].split()[0])
        lattice = [
            [float(item) * scale for item in lines[index].split()[:3]]
            for index in (2, 3, 4)
        ]
    except (IndexError, ValueError):
        return {}
    species_line = lines[5].split()
    count_line = lines[6].split()
    if not species_line or not count_line:
        return {"lattice_vectors": lattice}
    try:
        counts = [int(float(item)) for item in count_line]
    except ValueError:
        return {"lattice_vectors": lattice}
    species_counts = dict(zip(species_line, counts))
    return _drop_empty(
        {
            "comment": lines[0],
            "lattice_vectors": lattice,
            "species_counts": species_counts,
            "site_count": sum(counts),
        }
    )


def _parse_cif_scalars(text: str) -> dict[str, str]:
    scalars: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("_"):
            index += 1
            continue
        parts = line.split(maxsplit=1)
        key = parts[0]
        if len(parts) == 2:
            value = parts[1].strip()
            if value == ";":
                value, index = _read_cif_multiline(lines, index + 1)
            scalars[key] = value
        elif index + 1 < len(lines) and lines[index + 1].startswith(";"):
            value, index = _read_cif_multiline(lines, index + 2)
            scalars[key] = value
        index += 1
    return scalars


def _read_cif_multiline(lines: list[str], start: int) -> tuple[str, int]:
    payload: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if line.startswith(";"):
            return "\n".join(payload).strip(), index
        payload.append(line)
        index += 1
    return "\n".join(payload).strip(), index


def _parse_cif_atom_sites(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if not line.strip().lower().startswith("loop_"):
            continue
        headers: list[str] = []
        row_index = start + 1
        while row_index < len(lines):
            clean = lines[row_index].strip()
            if clean.startswith("_atom_site_"):
                headers.append(clean.split()[0])
                row_index += 1
                continue
            break
        if not headers or not any(
            header in headers
            for header in {"_atom_site_type_symbol", "_atom_site_label"}
        ):
            continue
        rows: list[list[str]] = []
        while row_index < len(lines):
            clean = lines[row_index].strip()
            if (
                not clean
                or clean.startswith("_")
                or clean.lower().startswith("loop_")
            ):
                break
            try:
                values = shlex.split(clean, posix=False)
            except ValueError:
                values = clean.split()
            if len(values) >= len(headers):
                rows.append(values[: len(headers)])
            row_index += 1
        return _summarize_atom_site_rows(headers, rows)
    return {}


def _summarize_atom_site_rows(
    headers: list[str],
    rows: list[list[str]],
) -> dict[str, Any]:
    if not rows:
        return {}
    element_index = (
        headers.index("_atom_site_type_symbol")
        if "_atom_site_type_symbol" in headers
        else headers.index("_atom_site_label")
    )
    elements: dict[str, int] = {}
    for row in rows:
        symbol = _element_from_atom_label(row[element_index])
        if symbol:
            elements[symbol] = elements.get(symbol, 0) + 1
    heavy_count = sum(
        count for symbol, count in elements.items() if symbol.upper() != "H"
    )
    return _drop_empty(
        {
            "site_count": len(rows),
            "heavy_atom_site_count": heavy_count,
            "element_site_counts": elements,
            "elements": sorted(elements),
        }
    )


def parse_formula_counts(formula: Any) -> dict[str, int]:
    """Return approximate element counts from a simple chemical
    formula."""

    text = _clean_cif_value(formula)
    if not text:
        return {}
    counts: dict[str, int] = {}
    for multiplier, segment in _formula_segments(text):
        for element, count_text in re.findall(
            r"([A-Z][a-z]?)([0-9.]*)", segment
        ):
            count = float(count_text) if count_text else 1.0
            total = multiplier * count
            counts[element] = counts.get(element, 0) + int(round(total))
    return dict(sorted((key, value) for key, value in counts.items() if value))


def _formula_segments(formula: str) -> list[tuple[float, str]]:
    cleaned = formula.replace("'", "").replace('"', "")
    cleaned = cleaned.replace(",", " ").replace(";", " ")
    segments: list[tuple[float, str]] = []
    for raw_segment in cleaned.split():
        match = re.fullmatch(r"([0-9.]+)?\(?([A-Za-z0-9]+)\)?", raw_segment)
        if not match:
            continue
        multiplier = float(match.group(1)) if match.group(1) else 1.0
        segments.append((multiplier, match.group(2)))
    return segments or [(1.0, re.sub(r"[^A-Za-z0-9]", "", cleaned))]


def _element_from_atom_label(value: str) -> str:
    clean = _clean_cif_value(value)
    match = re.match(r"([A-Za-z]{1,2})", clean)
    if not match:
        return ""
    symbol = match.group(1)
    return symbol[0].upper() + symbol[1:].lower()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_cif_value(value: Any) -> str:
    clean = _clean_text(value)
    if clean in {"?", "."}:
        return ""
    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
        clean = clean[1:-1]
    return clean.strip()


def _first_float(value: Any) -> float | None:
    clean = _clean_cif_value(value)
    if not clean:
        return None
    match = re.search(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", clean)
    if not match:
        return None
    return float(match.group(0))


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: _drop_empty(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
        return {
            key: item
            for key, item in cleaned.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            _drop_empty(item)
            for item in value
            if item not in (None, "", [], {})
        ]
    return value


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", value).strip("_")
    return cleaned or "structure.dat"


def _compact_tags(values: Iterable[str]) -> list[str]:
    tags: list[str] = []
    for value in values:
        clean = re.sub(r"[^A-Za-z0-9]+", "_", str(value).lower()).strip("_")
        if clean:
            tags.append(clean)
    return sorted(set(tags))
