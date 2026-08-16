import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from financial_ai.ml.transaction_classification.core.categories import ExpenseCategory

GENERATOR_VERSION = "controlled-english-training-generator-v1"
DEFAULT_RANDOM_SEED = 20260730
DEFAULT_EXAMPLES_PER_CATEGORY = 1_000
DEFAULT_OUTPUT_PATH = Path("data/runtime/ml/transaction_categories/english_training_v1.csv")

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
class EnglishCategoryProfile:
    merchants: tuple[str, ...]
    merchant_concepts: tuple[str, ...]
    details: tuple[str, ...]


PROFILES: dict[ExpenseCategory, EnglishCategoryProfile] = {
    ExpenseCategory.GROCERIES: EnglishCategoryProfile(
        merchants=(
            "Fresh Basket",
            "Pantry Lane",
            "Green Cart",
            "Family Market",
            "Daily Foods",
            "Home Grocer",
            "Harvest Shop",
            "Corner Produce",
        ),
        merchant_concepts=("Market Grove", "Food Basket", "Pantry Point", "Fresh Table"),
        details=(
            "weekly shop",
            "household groceries",
            "market purchase",
            "food supplies",
            "checkout receipt",
            "home provisions",
            "fresh produce",
            "daily essentials",
        ),
    ),
    ExpenseCategory.DINING: EnglishCategoryProfile(
        merchants=(
            "Copper Cafe",
            "Table House",
            "City Bistro",
            "Pizza Corner",
            "Lunch Room",
            "Noodle Kitchen",
            "River Restaurant",
            "Delivery Plate",
        ),
        merchant_concepts=("Dinner Table", "Meal Corner", "Cafe Garden", "Kitchen Door"),
        details=(
            "lunch order",
            "dinner bill",
            "takeaway meal",
            "table payment",
            "coffee and snack",
            "food delivery",
            "restaurant check",
            "canteen meal",
        ),
    ),
    ExpenseCategory.TRANSPORT: EnglishCategoryProfile(
        merchants=(
            "City Transit",
            "Metro Route",
            "Taxi Central",
            "Fuel Point",
            "Park Station",
            "Charge Lane",
            "Bike Share",
            "Regional Coach",
        ),
        merchant_concepts=("Urban Ride", "Travel Ground", "Road Link", "Move City"),
        details=(
            "transit ticket",
            "fuel purchase",
            "parking charge",
            "local ride",
            "charging station",
            "day pass",
            "taxi fare",
            "commuter route",
        ),
    ),
    ExpenseCategory.HOUSING: EnglishCategoryProfile(
        merchants=(
            "North Property",
            "Home Estate",
            "Tenant House",
            "Living Cooperative",
            "Park Management",
            "Student Residence",
            "Urban Housing",
            "South Apartments",
        ),
        merchant_concepts=("Rental Place", "Living Space", "Home Office", "Property Court"),
        details=(
            "monthly rent",
            "housing charge",
            "lease payment",
            "apartment rate",
            "residence fee",
            "parking space rent",
            "tenant payment",
            "property use",
        ),
    ),
    ExpenseCategory.UTILITIES: EnglishCategoryProfile(
        merchants=(
            "North Energy",
            "Clear Water",
            "City Gas",
            "Digital Cable",
            "Mobile Network",
            "Home Internet",
            "District Heat",
            "Waste Service",
        ),
        merchant_concepts=("Utility Grid", "Network Source", "Home Supply", "Service Line"),
        details=(
            "electric bill",
            "water service",
            "internet plan",
            "monthly utility",
            "gas account",
            "mobile service",
            "heating charge",
            "waste collection",
        ),
    ),
    ExpenseCategory.HEALTHCARE: EnglishCategoryProfile(
        merchants=(
            "Sunrise Pharmacy",
            "City Medical",
            "Park Dental",
            "Motion Therapy",
            "Health Supply",
            "North Laboratory",
            "Clear Vision",
            "Hearing Center",
        ),
        merchant_concepts=("Wellness Clinic", "Medical Point", "Care Practice", "Health Room"),
        details=(
            "medical visit",
            "prescription order",
            "dental treatment",
            "therapy session",
            "diagnostic test",
            "medical supplies",
            "patient charge",
            "health appointment",
        ),
    ),
    ExpenseCategory.SHOPPING: EnglishCategoryProfile(
        merchants=(
            "Modern Fashion",
            "Digital Store",
            "Home Interior",
            "Active Sports",
            "Care Retail",
            "Shoe Corner",
            "Household Shop",
            "Tool Market",
        ),
        merchant_concepts=("Retail Place", "Goods Market", "Favorite Item", "Shopping Lane"),
        details=(
            "retail purchase",
            "online order",
            "clothing item",
            "electronics",
            "household goods",
            "store checkout",
            "product shipment",
            "shopping cart",
        ),
    ),
    ExpenseCategory.ENTERTAINMENT: EnglishCategoryProfile(
        merchants=(
            "Light Cinema",
            "City Theater",
            "Sound Stream",
            "Game World",
            "Concert Hall",
            "Adventure Park",
            "Culture Museum",
            "Video Portal",
        ),
        merchant_concepts=("Leisure Place", "Media Room", "Stage Time", "Play Network"),
        details=(
            "admission ticket",
            "streaming plan",
            "game purchase",
            "event pass",
            "cinema booking",
            "digital media",
            "culture card",
            "recreation access",
        ),
    ),
    ExpenseCategory.TRAVEL: EnglishCategoryProfile(
        merchants=(
            "Morning Hotel",
            "Demo Airways",
            "Faraway Travel",
            "Holiday Home",
            "Coast Resort",
            "City Hostel",
            "Journey Portal",
            "Lake Camping",
        ),
        merchant_concepts=("Travel Path", "Vacation Place", "Guest House", "Journey Desk"),
        details=(
            "hotel booking",
            "flight ticket",
            "holiday package",
            "accommodation",
            "reservation deposit",
            "stay charge",
            "travel booking",
            "vacation rental",
        ),
    ),
    ExpenseCategory.INSURANCE: EnglishCategoryProfile(
        merchants=(
            "Secure Plus",
            "Future Cover",
            "Protection Direct",
            "Home Policy",
            "Motor Shield",
            "Health Cover",
            "Legal Guard",
            "Life Mutual",
        ),
        merchant_concepts=("Policy Office", "Safety Partner", "Cover Place", "Protection Plan"),
        details=(
            "insurance premium",
            "monthly policy",
            "coverage payment",
            "annual premium",
            "contract rate",
            "protection plan",
            "policy charge",
            "member premium",
        ),
    ),
    ExpenseCategory.EDUCATION: EnglishCategoryProfile(
        merchants=(
            "Learning Academy",
            "Language School",
            "City University",
            "Digital Training",
            "Music School",
            "Tutor Center",
            "Campus Books",
            "Knowledge Portal",
        ),
        merchant_concepts=("Study Path", "Course Place", "Campus Room", "Learning Point"),
        details=(
            "course fee",
            "tuition payment",
            "semester charge",
            "learning material",
            "training access",
            "exam fee",
            "class payment",
            "study subscription",
        ),
    ),
    ExpenseCategory.OTHER: EnglishCategoryProfile(
        merchants=(
            "Key Service",
            "Photo Studio",
            "Clean Textile",
            "Hair Workshop",
            "Pet Care",
            "Copy Center",
            "Parcel Regional",
            "Repair Help",
        ),
        merchant_concepts=("Everyday Help", "Service Point", "Task Works", "Handy Direct"),
        details=(
            "one time service",
            "repair work",
            "personal service",
            "general task",
            "small service",
            "work order",
            "service charge",
            "miscellaneous job",
        ),
    ),
}

MERCHANT_SUFFIXES = ("Group", "Regional")

FORMAT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "bracketed_debit": (
        "[Debit] {merchant} #{reference} {detail}",
        "[Debit] {merchant} {city} *{reference} {detail}",
    ),
    "web_id": (
        "[Debit] {merchant} WEB ID:{reference} {detail}",
        "[Debit] {merchant} {reference} WEB {detail}",
    ),
    "ppd_id": (
        "[Debit] {merchant} PPD ID:{reference} {detail}",
        "[Debit] ACH {merchant} {detail} PPD {reference}",
    ),
    "processor": (
        "[Debit] PP * {merchant} {detail} #{reference}",
        "[Debit] SQ * {merchant} {city} {detail}",
    ),
    "card_purchase": (
        "[Debit] CARD PURCHASE {merchant} {city} {reference}",
        "[Debit] POS {merchant} {detail} *{reference}",
    ),
    "recurring": (
        "[Debit] RECURRING {merchant} {detail} {reference}",
        "[Debit] AUTOPAY {merchant} ID:{reference} {detail}",
    ),
    "transfer": (
        "[Debit] ONLINE TRANSFER {merchant} {detail} {reference}",
        "[Debit] ACH PAYMENT TO {merchant} {reference} {detail}",
    ),
    "compact": (
        "[Debit] {merchant} {reference}",
        "[Debit] {merchant} {detail}",
    ),
}

CITIES = (
    "Austin",
    "Boston",
    "Chicago",
    "Denver",
    "Miami",
    "Phoenix",
    "Portland",
    "Raleigh",
    "Seattle",
    "Tampa",
)


def _partition(values: list[str], rng: random.Random) -> dict[str, list[str]]:
    shuffled = values.copy()
    rng.shuffle(shuffled)
    held_out = max(1, len(shuffled) // 8)
    train_end = len(shuffled) - 2 * held_out
    validation_end = len(shuffled) - held_out
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def _merchant_group(category: ExpenseCategory, merchant: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", merchant.casefold()).strip("_")
    return f"controlled_en_{category.value}_{normalized}"


def _merchants(category: ExpenseCategory) -> list[str]:
    profile = PROFILES[category]
    additional = [
        f"{concept} {suffix}"
        for concept in profile.merchant_concepts
        for suffix in MERCHANT_SUFFIXES
    ]
    return [*profile.merchants, *additional]


def _style(description: str, style_index: int) -> str:
    if style_index == 0:
        return description
    if style_index == 1:
        return description.lower()
    if style_index == 2:
        return description.replace("PAYMENT", "PMT").replace("PURCHASE", "PURCH")
    if style_index == 3:
        return description.replace(" ", "  ")
    return description[:59]


def generate_english_training_data_v1(
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    if examples_per_category < 8:
        raise ValueError("examples_per_category must be at least 8")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    format_rng = random.Random(f"{random_seed}:formats")
    format_partitions = _partition(list(FORMAT_TEMPLATES), format_rng)

    for category in ExpenseCategory:
        rng = random.Random(f"{random_seed}:{category.value}")
        merchants = _merchants(category)
        details = list(PROFILES[category].details)
        merchant_partitions = _partition(merchants, rng)
        detail_partitions = _partition(details, rng)
        held_out_rows = max(1, examples_per_category // 8)
        split_sizes = {
            "train": examples_per_category - 2 * held_out_rows,
            "validation": held_out_rows,
            "test": held_out_rows,
        }

        for split_name in SPLIT_NAMES:
            split_count = 0
            while split_count < split_sizes[split_name]:
                merchant = rng.choice(merchant_partitions[split_name])
                detail = rng.choice(detail_partitions[split_name])
                format_group = rng.choice(format_partitions[split_name])
                variants = FORMAT_TEMPLATES[format_group]
                variant_index = rng.randrange(len(variants))
                description = variants[variant_index].format(
                    merchant=merchant,
                    city=rng.choice(CITIES),
                    detail=detail,
                    reference=rng.randrange(100_000, 1_000_000),
                )
                description = _style(description, rng.randrange(5))
                if description in seen:
                    continue

                seen.add(description)
                split_count += 1
                detail_number = details.index(detail) + 1
                rows.append(
                    {
                        "example_id": f"en_v1_{category.value}_{split_name}_{split_count:04d}",
                        "description": description,
                        "target_category": category.value,
                        "language": "en",
                        "template_id": f"{format_group}_variant_{variant_index + 1:02d}",
                        "detail_group": f"{category.value}_detail_{detail_number:02d}",
                        "merchant_group": _merchant_group(category, merchant),
                        "format_group": format_group,
                        "split": split_name,
                    }
                )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_english_training_dataset_v1(
    destination: Path = DEFAULT_OUTPUT_PATH,
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[Path, Path]:
    data = generate_english_training_data_v1(examples_per_category, random_seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata_path = destination.with_suffix(".metadata.json")
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "random_seed": random_seed,
        "examples_per_category": examples_per_category,
        "row_count": len(data),
        "sha256": digest,
        "taxonomy_version": "transaction-categories-v1",
        "split_strategy": "disjoint-merchant-detail-format-v1",
        "legacy_train_rows_analyzed": 25_644,
        "legacy_patterns_used": [
            "bracketed debit prefixes",
            "numeric references",
            "WEB and PPD identifiers",
            "processor wrappers",
            "compact descriptions",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination, metadata_path


if __name__ == "__main__":
    dataset_path, metadata_path = write_english_training_dataset_v1()
    print(f"Dataset ready: {dataset_path}")
    print(f"Metadata ready: {metadata_path}")
