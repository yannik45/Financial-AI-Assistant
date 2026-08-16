import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from financial_ai.ml.transaction_classification.core.categories import ExpenseCategory

GENERATOR_VERSION = "german-training-generator-v1"
DEFAULT_RANDOM_SEED = 20260730
DEFAULT_EXAMPLES_PER_CATEGORY = 1_000
DEFAULT_OUTPUT_PATH = Path("data/runtime/ml/transaction_categories/german_training_v1.csv")

OUTPUT_COLUMNS = [
    "example_id",
    "description",
    "target_category",
    "language",
    "template_id",
    "merchant_group",
]

DESCRIPTION_TEMPLATES = (
    "KARTENZAHLUNG {merchant} {city} {detail} REF {reference}",
    "SEPA LASTSCHRIFT {merchant} {detail} MANDAT {reference}",
    "{merchant} {city} {detail} {payment} {reference}",
    "ZAHLUNG AN {merchant} VERWENDUNGSZWECK {detail} {reference}",
    "ONLINE BEZAHLT {merchant} {detail} AUFTRAG {reference}",
    "{payment} {merchant} BUCHUNGSTEXT {detail} {city}",
    "{merchant} RECHNUNG {reference} {detail}",
    "GIROPAY {merchant} {city} {detail} ID {reference}",
)

CITIES = (
    "AUGSBURG",
    "BIELEFELD",
    "DARMSTADT",
    "DRESDEN",
    "ERFURT",
    "FREIBURG",
    "HANNOVER",
    "KIEL",
    "MAINZ",
    "NUERNBERG",
    "POTSDAM",
    "ROSTOCK",
)

PAYMENT_TYPES = (
    "EC ZAHLUNG",
    "KONTAKTLOS",
    "ONLINEKAUF",
    "LASTSCHRIFT",
    "DAUERAUFTRAG",
)


@dataclass(frozen=True)
class CategoryProfile:
    merchants: tuple[str, ...]
    details: tuple[str, ...]


CATEGORY_PROFILES: dict[ExpenseCategory, CategoryProfile] = {
    ExpenseCategory.GROCERIES: CategoryProfile(
        merchants=(
            "FRISCHEMARKT NORD",
            "VORRATSKORB",
            "GRUENLAND SUPERMARKT",
            "MARKTHALLE WEST",
            "NAHKAUF DEMO",
            "LEBENSMITTELPUNKT",
            "BIOKORB SUED",
            "FAMILIENMARKT",
        ),
        details=("WOCHENEINKAUF", "LEBENSMITTEL", "SUPERMARKT", "HAUSHALTSWAREN"),
    ),
    ExpenseCategory.DINING: CategoryProfile(
        merchants=(
            "CAFE ABENDROT",
            "BISTRO STADTBLICK",
            "PIZZAOFEN DEMO",
            "RESTAURANT FLUSSUFER",
            "BACKSTUBE MORGEN",
            "NUDELBAR MITTE",
            "LIEFERKUECHE WEST",
            "KANTINE FORUM",
        ),
        details=("MITTAGESSEN", "ABENDESSEN", "GETRAENKE", "ESSENSBESTELLUNG"),
    ),
    ExpenseCategory.TRANSPORT: CategoryProfile(
        merchants=(
            "STADTMOBIL DEMO",
            "REGIOBUS NORD",
            "TAXIRUF ZENTRUM",
            "TANKPUNKT WEST",
            "PARKHAUS FORUM",
            "LADESTATION MOBIL",
            "RADLEIHE CITY",
            "BAHNREISE REGIONAL",
        ),
        details=("FAHRKARTE", "KRAFTSTOFF", "PARKGEBUEHR", "FAHRTKOSTEN"),
    ),
    ExpenseCategory.HOUSING: CategoryProfile(
        merchants=(
            "WOHNRAUM VERWALTUNG NORD",
            "MIETVEREIN SONNENSEITE",
            "HAUSKONTOR DEMO",
            "WOHNGENOSSENSCHAFT WEST",
            "IMMOBILIENSERVICE PARK",
            "STUDENTENWOHNEN MITTE",
            "WOHNSTIFTUNG ELBBLICK",
            "MIETOBJEKT VERWALTUNG SUED",
        ),
        details=("MONATSMIETE", "WOHNRAUMNUTZUNG", "MIETRATE", "STELLPLATZMIETE"),
    ),
    ExpenseCategory.UTILITIES: CategoryProfile(
        merchants=(
            "ENERGIEWERK NORD",
            "WASSERVERSORGUNG DEMO",
            "STADTGAS MITTE",
            "NETZKABEL DIGITAL",
            "MOBILFUNK DIREKT",
            "INTERNETWERK SUED",
            "WAERMEVERSORGUNG WEST",
            "ABFALLSERVICE REGION",
        ),
        details=("STROMABSCHLAG", "WASSERRECHNUNG", "INTERNETTARIF", "MONATSABSCHLAG"),
    ),
    ExpenseCategory.HEALTHCARE: CategoryProfile(
        merchants=(
            "APOTHEKE SONNENWEG",
            "ARZTPRAXIS AM TOR",
            "ZAHNMEDIZIN DEMO",
            "THERAPIEZENTRUM VITAL",
            "SANITAETSHAUS AKTIV",
            "LABORPRAXIS NORD",
            "AUGENZENTRUM KLARBLICK",
            "HOERAKUSTIK FORUM",
        ),
        details=("BEHANDLUNG", "MEDIKAMENTE", "THERAPIE", "MEDIZINISCHER EIGENANTEIL"),
    ),
    ExpenseCategory.SHOPPING: CategoryProfile(
        merchants=(
            "MODEHAUS LINIE",
            "TECHNIKMARKT DEMO",
            "WOHNWERK EINRICHTUNG",
            "SPORTLADEN AKTIV",
            "DROGERIE GLANZ",
            "SCHUHPUNKT MITTE",
            "HAUSHALTSWAREN NORD",
            "WERKZEUGMARKT PROJEKT",
        ),
        details=("WARENEINKAUF", "ONLINEBESTELLUNG", "KLEIDUNG", "ELEKTRONIK"),
    ),
    ExpenseCategory.ENTERTAINMENT: CategoryProfile(
        merchants=(
            "KINO LICHTSPIEL",
            "THEATERBUEHNE DEMO",
            "MUSIKSTREAM KLANG",
            "SPIELEWELT ONLINE",
            "KONZERTHAUS FORUM",
            "FREIZEITPARK ABENTEUER",
            "MUSEUMSKARTE REGION",
            "VIDEOPORTAL PLUS",
        ),
        details=("EINTRITTSKARTE", "STREAMINGABO", "SPIELEKAUF", "VERANSTALTUNGSTICKET"),
    ),
    ExpenseCategory.TRAVEL: CategoryProfile(
        merchants=(
            "HOTEL MORGENSONNE",
            "FLUGLINIE DEMO",
            "REISEBUERO FERNWEH",
            "FERIENHAUS PORTAL",
            "URLAUBSPARK KUESTE",
            "HOSTEL STADTTOR",
            "REISEPORTAL WEITBLICK",
            "CAMPINGPLATZ SEEBLICK",
        ),
        details=("HOTELBUCHUNG", "FLUGTICKET", "URLAUBSREISE", "UNTERKUNFT"),
    ),
    ExpenseCategory.INSURANCE: CategoryProfile(
        merchants=(
            "SICHERPLUS VERSICHERUNG",
            "VORSORGEWERK DEMO",
            "SCHUTZBRIEF DIREKT",
            "HAUSRAT SICHER",
            "MOBILPOLICE NORD",
            "GESUNDHEITSSCHUTZ PLUS",
            "RECHTSSCHUTZ PARTNER",
            "LEBENSSCHUTZ VEREIN",
        ),
        details=("VERSICHERUNGSBEITRAG", "MONATSPRAEMIE", "POLICENRATE", "JAHRESBEITRAG"),
    ),
    ExpenseCategory.EDUCATION: CategoryProfile(
        merchants=(
            "LERNAKADEMIE DEMO",
            "SPRACHSCHULE DIALOG",
            "HOCHSCHULE FORUM",
            "WEITERBILDUNG DIGITAL",
            "MUSIKSCHULE TAKT",
            "NACHHILFEWERK",
            "FACHBUCH CAMPUS",
            "KURSPORTAL WISSEN",
        ),
        details=("KURSGEBUEHR", "UNTERRICHT", "SEMESTERBEITRAG", "LEHRMATERIAL"),
    ),
    ExpenseCategory.OTHER: CategoryProfile(
        merchants=(
            "SCHLUESSELSERVICE DEMO",
            "FOTOSTUDIO MOMENT",
            "TEXTILPFLEGE SAUBER",
            "FRISEURWERK SCHNITT",
            "TIERPFLEGE PFOTE",
            "KOPIERSERVICE FORUM",
            "PAKETSERVICE REGIONAL",
            "HANDWERKSHILFE DIREKT",
        ),
        details=("DIENSTLEISTUNG", "REPARATUR", "PERSOENLICHER SERVICE", "SONSTIGER AUFTRAG"),
    ),
}


def _merchant_group(category: ExpenseCategory, merchant: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", merchant.casefold()).strip("_")
    return f"generated_{category.value}_{normalized}"


def generate_german_training_data(
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    if examples_per_category <= 0:
        raise ValueError("examples_per_category must be positive")

    random_generator = random.Random(random_seed)
    rows: list[dict[str, str]] = []
    seen_descriptions: set[str] = set()

    for category in ExpenseCategory:
        profile = CATEGORY_PROFILES[category]
        category_count = 0

        while category_count < examples_per_category:
            merchant = random_generator.choice(profile.merchants)
            detail = random_generator.choice(profile.details)
            city = random_generator.choice(CITIES)
            payment = random_generator.choice(PAYMENT_TYPES)
            template_index = random_generator.randrange(len(DESCRIPTION_TEMPLATES))
            reference = f"{random_generator.randrange(100_000, 1_000_000)}"
            description = DESCRIPTION_TEMPLATES[template_index].format(
                merchant=merchant,
                city=city,
                detail=detail,
                payment=payment,
                reference=reference,
            )

            if description in seen_descriptions:
                continue

            seen_descriptions.add(description)
            category_count += 1
            rows.append(
                {
                    "example_id": f"de_train_{category.value}_{category_count:04d}",
                    "description": description,
                    "target_category": category.value,
                    "language": "de",
                    "template_id": f"template_{template_index + 1:02d}",
                    "merchant_group": _merchant_group(category, merchant),
                }
            )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_german_training_dataset(
    destination: Path = DEFAULT_OUTPUT_PATH,
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[Path, Path]:
    training_data = generate_german_training_data(
        examples_per_category=examples_per_category,
        random_seed=random_seed,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(destination, index=False)

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata_path = destination.with_suffix(".metadata.json")
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "random_seed": random_seed,
        "examples_per_category": examples_per_category,
        "row_count": len(training_data),
        "sha256": digest,
        "taxonomy_version": "transaction-categories-v1",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination, metadata_path


if __name__ == "__main__":
    dataset_path, metadata_path = write_german_training_dataset()
    print(f"Dataset ready: {dataset_path}")
    print(f"Metadata ready: {metadata_path}")
