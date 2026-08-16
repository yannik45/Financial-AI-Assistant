from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from random import Random


@dataclass(frozen=True)
class DemoBankTransaction:
    booked_at: date
    description: str
    amount: Decimal
    transaction_type: str
    counterparty: str | None
    expected_category: str


@dataclass(frozen=True)
class TransactionTemplate:
    descriptions: tuple[str, ...]
    counterparties: tuple[str, ...]
    amount_range: tuple[Decimal, Decimal]
    transaction_type: str
    category: str


VARIABLE_TEMPLATES = (
    TransactionTemplate(
        ("VISA EDEKA SAGT DANKE", "Kartenzahlung REWE MARKT", "LIDL DIENSTLEISTUNG"),
        ("EDEKA", "REWE", "LIDL"),
        (Decimal("18.00"), Decimal("115.00")),
        "card_payment",
        "groceries",
    ),
    TransactionTemplate(
        ("VISA DB BAHN AUTOMAT", "BVG TICKET APP", "Kartenzahlung SHELL STATION"),
        ("Deutsche Bahn", "BVG", "Shell"),
        (Decimal("3.20"), Decimal("79.00")),
        "card_payment",
        "transport",
    ),
    TransactionTemplate(
        ("Kartenzahlung CAFE CENTRAL", "VISA MAMMA MIA", "SUMUP RESTAURANT KREUZBERG"),
        ("Cafe Central", "Mamma Mia", "Restaurant Kreuzberg"),
        (Decimal("4.00"), Decimal("72.00")),
        "card_payment",
        "dining",
    ),
    TransactionTemplate(
        ("VISA ZARA DE", "PAYPAL *ABOUT YOU", "Kartenzahlung UNIQLO"),
        ("Zara", "About You", "Uniqlo"),
        (Decimal("19.00"), Decimal("145.00")),
        "card_payment",
        "shopping",
    ),
    TransactionTemplate(
        ("SEPA LASTSCHRIFT NETFLIX.COM", "PAYPAL *SPOTIFY", "VISA KINO INTERNATIONAL"),
        ("Netflix", "Spotify", "Kino International"),
        (Decimal("8.00"), Decimal("28.00")),
        "direct_debit",
        "entertainment",
    ),
    TransactionTemplate(
        ("APOTHEKE AM MARKT", "DOCTOLIB GMBH", "VISA DM DROGERIE"),
        ("Apotheke am Markt", "Doctolib", "dm"),
        (Decimal("7.00"), Decimal("85.00")),
        "card_payment",
        "healthcare",
    ),
)


def _amount(rng: Random, bounds: tuple[Decimal, Decimal]) -> Decimal:
    minimum, maximum = bounds
    cents = rng.randint(int(minimum * 100), int(maximum * 100))
    return Decimal(cents).scaleb(-2)


def generate_demo_bank_feed(
    *, seed: int, year: int, month: int, variable_count: int = 12
) -> list[DemoBankTransaction]:
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    if not 1 <= variable_count <= 40:
        raise ValueError("variable_count must be between 1 and 40")

    rng = Random(seed)
    last_day = monthrange(year, month)[1]
    transactions = [
        DemoBankTransaction(
            date(year, month, min(2, last_day)),
            rng.choice(("GEHALT DEMO DIGITAL GMBH", "LOHN/GEHALT DEMO LABS SEPA")),
            Decimal("3200.00"),
            "salary",
            "Demo Digital GmbH",
            "income",
        ),
        DemoBankTransaction(
            date(year, month, min(4, last_day)),
            rng.choice(("SEPA DAUERAUFTRAG MIETE", "WOHNRAUMMIETE VERTRAG 4711")),
            Decimal("-1120.00"),
            "direct_debit",
            "Demo Hausverwaltung GmbH",
            "housing",
        ),
        DemoBankTransaction(
            date(year, month, min(8, last_day)),
            rng.choice(("SEPA LASTSCHRIFT STROM", "ABSCHLAG ENERGIE KUNDENNR 2048")),
            Decimal("-76.00"),
            "direct_debit",
            "Demo Energie AG",
            "utilities",
        ),
    ]

    for _ in range(variable_count):
        template = rng.choice(VARIABLE_TEMPLATES)
        transactions.append(
            DemoBankTransaction(
                booked_at=date(year, month, rng.randint(1, last_day)),
                description=rng.choice(template.descriptions),
                amount=-_amount(rng, template.amount_range),
                transaction_type=template.transaction_type,
                counterparty=rng.choice(template.counterparties),
                expected_category=template.category,
            )
        )

    return sorted(transactions, key=lambda item: (item.booked_at, item.description))
