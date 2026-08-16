import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from financial_ai.ml.transaction_classification.core.categories import ExpenseCategory
from financial_ai.ml.transaction_classification.data.german_training_generator import (
    CATEGORY_PROFILES,
    CITIES,
    PAYMENT_TYPES,
    _merchant_group,
)

GENERATOR_VERSION = "german-training-generator-v2"
DEFAULT_RANDOM_SEED = 20260730
DEFAULT_EXAMPLES_PER_CATEGORY = 1_000
DEFAULT_OUTPUT_PATH = Path("data/runtime/ml/transaction_categories/german_training_v2.csv")

OUTPUT_COLUMNS = [
    "example_id",
    "description",
    "target_category",
    "language",
    "template_id",
    "detail_group",
    "merchant_group",
    "format_group",
    "split",
]

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class VersionTwoProfile:
    merchant_concepts: tuple[str, ...]
    additional_details: tuple[str, ...]


VERSION_TWO_PROFILES: dict[ExpenseCategory, VersionTwoProfile] = {
    ExpenseCategory.GROCERIES: VersionTwoProfile(
        merchant_concepts=("VITALKOST", "HEIMATKORB", "TAGESFRISCH", "VORRATSWELT"),
        additional_details=(
            "EINKAUF DES TAEGLICHEN BEDARFS",
            "MARKTEINKAUF",
            "KASSENBON",
            "VORRAETE",
        ),
    ),
    ExpenseCategory.DINING: VersionTwoProfile(
        merchant_concepts=("TELLERZEIT", "GENUSSECK", "PAUSENKUECHE", "SPEISEWERK"),
        additional_details=(
            "TISCHRECHNUNG",
            "BESTELLUNG ZUM MITNEHMEN",
            "BEWIRTUNG",
            "SNACK UND GETRAENK",
        ),
    ),
    ExpenseCategory.TRANSPORT: VersionTwoProfile(
        merchant_concepts=("FAHRPUNKT", "WEGWERK", "MOBILROUTE", "STADTFAHRT"),
        additional_details=("TAGESPASS", "LADESAEULE", "STELLPLATZ", "WEGESTRECKE"),
    ),
    ExpenseCategory.HOUSING: VersionTwoProfile(
        merchant_concepts=("WOHNKONTOR", "RAUMWERK", "MIETPUNKT", "HAUSRAUM"),
        additional_details=("NUTZUNGSENTGELT", "WOHNKOSTENRATE", "OBJEKTMIETE", "WOHNHEIMBEITRAG"),
    ),
    ExpenseCategory.UTILITIES: VersionTwoProfile(
        merchant_concepts=("NETZQUELLE", "VERSORGUNGSWERK", "HAUSNETZ", "LEITUNGSWELT"),
        additional_details=(
            "GRUNDVERSORGUNG",
            "ANSCHLUSSTARIF",
            "VERBRAUCHSRATE",
            "VERSORGUNGSABSCHLAG",
        ),
    ),
    ExpenseCategory.HEALTHCARE: VersionTwoProfile(
        merchant_concepts=("GESUNDRAUM", "PRAXISWERK", "VITALHILFE", "MEDPUNKT"),
        additional_details=("SPRECHSTUNDE", "REZEPTBESTELLUNG", "HEILBEHANDLUNG", "DIAGNOSTIK"),
    ),
    ExpenseCategory.SHOPPING: VersionTwoProfile(
        merchant_concepts=("KAUFRAUM", "WARENWELT", "LIEBLINGSSTUECK", "MARKTPLATZ DIREKT"),
        additional_details=("VERSANDKAUF", "FILIALEINKAUF", "ARTIKELBESTELLUNG", "WARENKORB"),
    ),
    ExpenseCategory.ENTERTAINMENT: VersionTwoProfile(
        merchant_concepts=("FREIZEITWELT", "KLANGRAUM", "BUEHNENZEIT", "SPIELPORTAL"),
        additional_details=("FREIZEITPASS", "MEDIENZUGANG", "KULTURKARTE", "DIGITALER SPIELINHALT"),
    ),
    ExpenseCategory.TRAVEL: VersionTwoProfile(
        merchant_concepts=("FERNWEG", "URLAUBSRAUM", "REISEPFAD", "GASTHAUS PORTAL"),
        additional_details=(
            "RESERVIERUNGSANZAHLUNG",
            "AUFENTHALTSKOSTEN",
            "REISEPAKET",
            "FERNREISEBUCHUNG",
        ),
    ),
    ExpenseCategory.INSURANCE: VersionTwoProfile(
        merchant_concepts=("VORSORGEKONTOR", "POLICENWERK", "SICHERRAUM", "SCHUTZPARTNER"),
        additional_details=("VERTRAGSRATE", "SCHUTZPAKET", "VORSORGEBEITRAG", "DECKUNGSBEITRAG"),
    ),
    ExpenseCategory.EDUCATION: VersionTwoProfile(
        merchant_concepts=("WISSENSRAUM", "LERNPFAD", "CAMPUSWERK", "KURSAKADEMIE"),
        additional_details=("FORTBILDUNG", "PRUEFUNGSGEBUEHR", "LERNZUGANG", "SCHULUNGSRECHNUNG"),
    ),
    ExpenseCategory.OTHER: VersionTwoProfile(
        merchant_concepts=("ALLTAGSHILFE", "SERVICEPUNKT", "AUFTRAGSWERK", "HANDGRIFF DIREKT"),
        additional_details=(
            "EINMALIGER AUFTRAG",
            "ARBEITSLEISTUNG",
            "KLEINSERVICE",
            "AUFTRAGSENTGELT",
        ),
    ),
}

MERCHANT_SUFFIXES = ("KONTOR", "REGIONAL")

FORMAT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "card": (
        "KARTENZAHLUNG {merchant} {city} {detail} REF {reference}",
        "EC {merchant} {detail} {city} NR {reference}",
    ),
    "sepa": (
        "SEPA LASTSCHRIFT {merchant} MANDAT {reference} {detail}",
        "SEPA-LS {merchant} {detail} EREF {reference}",
    ),
    "transfer": (
        "UEBERWEISUNG AN {merchant} ZWECK {detail} {reference}",
        "ONLINE UEBERW {merchant} {reference} {detail}",
    ),
    "online": (
        "ONLINE BEZAHLT {merchant} {detail} AUFTRAG {reference}",
        "WEBKAUF {merchant} ID {reference} {detail}",
    ),
    "invoice": (
        "RECHNUNG {reference} {merchant} {detail}",
        "RG {reference} VON {merchant} {detail}",
    ),
    "standing_order": (
        "DAUERAUFTRAG {merchant} {detail} {reference}",
        "DA AUFTR {merchant} {reference} {detail}",
    ),
    "processor": (
        "ZAHLUNGSDIENST * {merchant} {detail} {reference}",
        "PAYMENT WRAPPER {merchant} {reference} {detail}",
    ),
    "compact": (
        "{merchant} {city} {detail} {reference}",
        "{payment} {merchant} {detail}",
    ),
}


def _partition_groups(
    values: list[str],
    random_generator: random.Random,
) -> dict[str, list[str]]:
    shuffled_values = values.copy()
    random_generator.shuffle(shuffled_values)
    held_out_count = max(1, len(shuffled_values) // 8)
    train_end = len(shuffled_values) - 2 * held_out_count
    validation_end = len(shuffled_values) - held_out_count
    return {
        "train": shuffled_values[:train_end],
        "validation": shuffled_values[train_end:validation_end],
        "test": shuffled_values[validation_end:],
    }


def _apply_text_style(description: str, style_index: int) -> str:
    if style_index == 0:
        return description.upper()
    if style_index == 1:
        return description.lower()
    if style_index == 2:
        return description.title()
    if style_index == 3:
        return description.upper().replace("ZAHLUNG", "ZAHLG")
    return "  ".join(description.upper().split())


def _build_merchants(category: ExpenseCategory) -> list[str]:
    additional_merchants = [
        f"{concept} {suffix}"
        for concept in VERSION_TWO_PROFILES[category].merchant_concepts
        for suffix in MERCHANT_SUFFIXES
    ]
    return [*CATEGORY_PROFILES[category].merchants, *additional_merchants]


def _build_details(category: ExpenseCategory) -> list[str]:
    return [
        *CATEGORY_PROFILES[category].details,
        *VERSION_TWO_PROFILES[category].additional_details,
    ]


def generate_german_training_data_v2(
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    if examples_per_category < 8:
        raise ValueError("examples_per_category must be at least 8")

    rows: list[dict[str, str]] = []
    seen_descriptions: set[str] = set()
    format_random_generator = random.Random(f"{random_seed}:formats")
    format_partitions = _partition_groups(
        list(FORMAT_TEMPLATES),
        format_random_generator,
    )

    for category in ExpenseCategory:
        random_generator = random.Random(f"{random_seed}:{category.value}")
        merchants = _build_merchants(category)
        details = _build_details(category)

        merchant_partitions = _partition_groups(merchants, random_generator)
        detail_partitions = _partition_groups(details, random_generator)

        held_out_rows = max(1, examples_per_category // 8)
        split_sizes = {
            "train": examples_per_category - 2 * held_out_rows,
            "validation": held_out_rows,
            "test": held_out_rows,
        }

        for split_name in SPLIT_NAMES:
            split_count = 0
            while split_count < split_sizes[split_name]:
                merchant = random_generator.choice(merchant_partitions[split_name])
                detail = random_generator.choice(detail_partitions[split_name])
                format_group = random_generator.choice(format_partitions[split_name])
                template_variants = FORMAT_TEMPLATES[format_group]
                template_index = random_generator.randrange(len(template_variants))
                template = template_variants[template_index]
                reference = f"{random_generator.randrange(100_000, 1_000_000)}"
                description = template.format(
                    merchant=merchant,
                    city=random_generator.choice(CITIES),
                    detail=detail,
                    payment=random_generator.choice(PAYMENT_TYPES),
                    reference=reference,
                )
                description = _apply_text_style(
                    description,
                    random_generator.randrange(5),
                )

                if description in seen_descriptions:
                    continue

                seen_descriptions.add(description)
                split_count += 1
                detail_number = details.index(detail) + 1
                rows.append(
                    {
                        "example_id": (f"de_v2_{category.value}_{split_name}_{split_count:04d}"),
                        "description": description,
                        "target_category": category.value,
                        "language": "de",
                        "template_id": (f"{format_group}_variant_{template_index + 1:02d}"),
                        "detail_group": (f"{category.value}_detail_{detail_number:02d}"),
                        "merchant_group": _merchant_group(category, merchant),
                        "format_group": format_group,
                        "split": split_name,
                    }
                )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_german_training_dataset_v2(
    destination: Path = DEFAULT_OUTPUT_PATH,
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[Path, Path]:
    training_data = generate_german_training_data_v2(
        examples_per_category=examples_per_category,
        random_seed=random_seed,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(destination, index=False)

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata_path = destination.with_suffix(".metadata.json")
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "base_generator_version": "german-training-generator-v1",
        "random_seed": random_seed,
        "examples_per_category": examples_per_category,
        "row_count": len(training_data),
        "sha256": digest,
        "taxonomy_version": "transaction-categories-v1",
        "split_strategy": "disjoint-merchant-detail-format-v1",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination, metadata_path


if __name__ == "__main__":
    dataset_path, metadata_path = write_german_training_dataset_v2()
    print(f"Dataset ready: {dataset_path}")
    print(f"Metadata ready: {metadata_path}")
