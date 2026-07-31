import hashlib
import json
from pathlib import Path

import pandas as pd

from financial_ai.ml.categories import ExpenseCategory
from financial_ai.ml.transaction_classification import TransactionCategory

CHALLENGE_VERSION = "text-classification-challenge-v2"
V1_CHALLENGE_PATH = Path(
    "data/evaluation/transaction_categories/text_classification_challenge_v1.csv"
)
V1_METADATA_PATH = Path(
    "data/evaluation/transaction_categories/text_classification_challenge_v1.metadata.json"
)
DEFAULT_CHALLENGE_PATH = Path(
    "data/evaluation/transaction_categories/text_classification_challenge_v2.csv"
)
DEFAULT_METADATA_PATH = Path(
    "data/evaluation/transaction_categories/text_classification_challenge_v2.metadata.json"
)
EXPECTED_COLUMNS = [
    "id",
    "description",
    "counterparty",
    "amount",
    "expected_category",
    "language",
    "difficulty",
    "ambiguity",
    "scenario_group",
    "notes",
]

# These phrases were authored as a frozen product challenge, not generated from
# training rows. Known development examples live in tests and are excluded here.
CURATED_TEXTS: dict[str, dict[str, tuple[str, ...]]] = {
    "income": {
        "en": (
            "Monthly payroll credit",
            "Employer compensation July",
            "Wages ACME Ltd",
            "Regular work payment",
            "Annual bonus received",
            "Interest credit from bank",
            "Incoming freelance earnings",
        ),
        "de": (
            "Monatliche Gehaltsgutschrift",
            "Arbeitgeber Zahlung Juli",
            "Lohn Firma Beispiel",
            "Vergütung für Arbeit",
            "Jahresbonus erhalten",
            "Zinsen von der Bank",
            "Einnahmen freiberufliche Arbeit",
        ),
    },
    "investments": {
        "en": (
            "Quarterly dividend credit",
            "Broker securities purchase",
            "ETF purchase settlement",
            "Shareholder distribution",
            "Portfolio custody booking",
            "Bought index fund units",
            "Capital markets account movement",
        ),
        "de": (
            "Quartalsdividende Gutschrift",
            "Broker Wertpapierkauf",
            "ETF Kauf Abrechnung",
            "Ausschüttung an Anteilseigner",
            "Depotbuchung Kapitalanlage",
            "Indexfonds Anteile gekauft",
            "Kapitalmarkt Konto Bewegung",
        ),
    },
    "fees": {
        "en": (
            "Monthly bank fee",
            "Account service charge",
            "Card replacement fee",
            "Processing cost debit",
            "Current account administration",
            "Foreign payment surcharge",
            "Small banking charge",
        ),
        "de": (
            "Monatliche Bankgebühr",
            "Kontoführungsgebühr",
            "Gebühr Ersatzkarte",
            "Bearbeitungskosten Abbuchung",
            "Verwaltung Girokonto",
            "Auslandseinsatzentgelt",
            "Kleine Bankkosten",
        ),
    },
    "taxes": {
        "en": (
            "Income tax payment",
            "Tax office direct debit",
            "Annual tax settlement",
            "Government revenue payment",
            "Municipal levy",
            "Fiscal authority debit",
            "Quarterly advance assessment",
        ),
        "de": (
            "Einkommensteuer Zahlung",
            "Finanzamt Lastschrift",
            "Jährliche Steuerabrechnung",
            "Zahlung an Steuerbehörde",
            "Kommunale Abgabe",
            "Fiskus Abbuchung",
            "Vierteljährliche Vorauszahlung",
        ),
    },
    "savings": {
        "en": (
            "Transfer to savings account",
            "Monthly savings contribution",
            "Emergency fund deposit",
            "Move money to rainy day fund",
            "Reserve account allocation",
            "Personal nest egg",
            "Long term cash reserve",
        ),
        "de": (
            "Übertrag auf Sparkonto",
            "Monatliche Sparrate",
            "Einzahlung Notgroschen",
            "Geld auf Tagesgeld",
            "Zuweisung Rücklagenkonto",
            "Persönliche Rücklage",
            "Langfristige Barreserve",
        ),
    },
    "cash": {
        "en": (
            "ATM cash withdrawal station",
            "Cash machine city center",
            "Banknote withdrawal terminal",
            "Money taken from cashpoint",
            "Physical cash withdrawal",
            "Automated teller debit",
            "Cash obtained near station",
        ),
        "de": (
            "Bargeldabhebung Geldautomat",
            "Geldautomat Innenstadt",
            "Banknoten am Automaten",
            "Bargeld am Terminal geholt",
            "Auszahlung in bar",
            "Automatenverfügung Bahnhof",
            "Bares nahe Hauptbahnhof",
        ),
    },
    "groceries": {
        "en": (
            "Grocery basket Green Market",
            "Weekly supermarket purchase",
            "Fresh food and household supplies",
            "Neighborhood produce store",
            "Family pantry restock",
            "Everyday food shop",
            "Market basket Saturday",
        ),
        "de": (
            "Lebensmitteleinkauf Grüner Markt",
            "Wöchentlicher Supermarkt Einkauf",
            "Frische Waren und Haushaltsbedarf",
            "Obstladen im Viertel",
            "Vorräte für die Familie",
            "Einkauf für den Alltag",
            "Marktkorb Samstag",
        ),
    },
    "dining": {
        "en": (
            "Dinner at Riverside Restaurant",
            "Coffee and cake downtown",
            "Pizza delivery evening",
            "Lunch counter payment",
            "Morning bakery visit",
            "Takeaway meal order",
            "Food truck at work",
        ),
        "de": (
            "Abendessen Restaurant am Fluss",
            "Kaffee und Kuchen Innenstadt",
            "Pizza Lieferung abends",
            "Mittagessen Kantine",
            "Besuch in der Bäckerei",
            "Essen zum Mitnehmen",
            "Imbisswagen bei der Arbeit",
        ),
    },
    "transport": {
        "en": (
            "City transit monthly ticket",
            "Fuel at North Road station",
            "Parking garage center",
            "Taxi ride home",
            "Regional train ticket",
            "Electric vehicle charging",
            "Commuter mobility payment",
        ),
        "de": (
            "Monatskarte Nahverkehr",
            "Tanken an der Nordstraße",
            "Parkhaus Innenstadt",
            "Taxifahrt nach Hause",
            "Regionalbahn Fahrkarte",
            "Elektroauto Ladestation",
            "Zahlung für Arbeitsweg",
        ),
    },
    "housing": {
        "en": (
            "Apartment rent August",
            "Mortgage installment home",
            "Property manager monthly debit",
            "Housing cooperative contribution",
            "Payment for my flat",
            "Residential lease charge",
            "Landlord standing order",
        ),
        "de": (
            "Wohnungsmiete August",
            "Hypothekenrate Eigenheim",
            "Hausverwaltung monatliche Abbuchung",
            "Beitrag Wohnungsgenossenschaft",
            "Zahlung für meine Wohnung",
            "Mietvertrag Belastung",
            "Dauerauftrag Vermieter",
        ),
    },
    "utilities": {
        "en": (
            "Electricity provider monthly bill",
            "Home internet contract",
            "Water utility charge",
            "Mobile phone plan",
            "Natural gas account debit",
            "Broadband service payment",
            "Household energy adjustment",
        ),
        "de": (
            "Stromanbieter Monatsrechnung",
            "Internetvertrag Zuhause",
            "Wasserwerk Abschlag",
            "Mobilfunktarif",
            "Gasanbieter Lastschrift",
            "Breitband Anschluss Zahlung",
            "Energie Nachzahlung Haushalt",
        ),
    },
    "healthcare": {
        "en": (
            "Pharmacy prescription purchase",
            "Dental practice treatment",
            "City clinic invoice",
            "Doctor consultation fee",
            "Physiotherapy appointment",
            "Medical laboratory service",
            "Urgent care visit",
        ),
        "de": (
            "Apotheke Rezeptkauf",
            "Zahnarzt Behandlung",
            "Klinik Rechnung",
            "Arztbesuch Gebühr",
            "Physiotherapie Termin",
            "Medizinisches Labor",
            "Ärztlicher Bereitschaftsdienst",
        ),
    },
    "shopping": {
        "en": (
            "Clothing order online",
            "Electronics store purchase",
            "Large marketplace order",
            "New household appliance",
            "Department store checkout",
            "Shoes from local retailer",
            "General merchandise purchase",
        ),
        "de": (
            "Kleidung online bestellt",
            "Elektronikmarkt Einkauf",
            "Große Marktplatz Bestellung",
            "Neues Haushaltsgerät",
            "Kaufhaus Kasse",
            "Schuhe vom Händler",
            "Allgemeiner Warenkauf",
        ),
    },
    "entertainment": {
        "en": (
            "Cinema tickets Friday",
            "Streaming monthly subscription",
            "Concert venue booking",
            "Video game purchase",
            "Museum admission",
            "Theatre evening",
            "Recreation membership",
        ),
        "de": (
            "Kinokarten Freitag",
            "Streaming Monatsabo",
            "Konzert Eintritt",
            "Videospiel gekauft",
            "Museum Eintritt",
            "Theaterabend",
            "Freizeit Mitgliedschaft",
        ),
    },
    "travel": {
        "en": (
            "Hotel reservation Barcelona",
            "Airline flight booking",
            "Holiday apartment deposit",
            "Travel agency package",
            "Airport accommodation",
            "Overnight stay abroad",
            "Vacation booking portal",
        ),
        "de": (
            "Hotelreservierung Barcelona",
            "Fluggesellschaft Buchung",
            "Ferienwohnung Anzahlung",
            "Reisebüro Pauschalreise",
            "Unterkunft am Flughafen",
            "Übernachtung im Ausland",
            "Urlaub Buchungsportal",
        ),
    },
    "insurance": {
        "en": (
            "Liability insurance premium",
            "Vehicle policy monthly",
            "Home contents coverage",
            "Health insurer contribution",
            "Travel protection policy",
            "Annual protection contract",
            "Insurer recurring debit",
        ),
        "de": (
            "Haftpflichtversicherung Beitrag",
            "Kfz Versicherung monatlich",
            "Hausrat Absicherung",
            "Krankenkasse Beitrag",
            "Reiseversicherung Police",
            "Jährlicher Schutzvertrag",
            "Versicherer Lastschrift",
        ),
    },
    "education": {
        "en": (
            "University tuition installment",
            "Online course enrollment",
            "Language school payment",
            "Professional training seminar",
            "Exam registration charge",
            "Learning platform subscription",
            "Continuing education workshop",
        ),
        "de": (
            "Universität Studiengebühr",
            "Onlinekurs Anmeldung",
            "Sprachschule Zahlung",
            "Berufliche Weiterbildung Seminar",
            "Prüfungsanmeldung Gebühr",
            "Lernplattform Abonnement",
            "Fortbildung Workshop",
        ),
    },
    "other": {
        "en": (
            "Miscellaneous local service",
            "Unknown merchant reference 482",
            "General payment reference",
            "Community contribution",
            "Private transfer to Alex",
            "One-off personal service",
            "Unclear recurring debit",
        ),
        "de": (
            "Sonstige lokale Dienstleistung",
            "Unbekannter Händler Referenz 482",
            "Allgemeine Zahlung",
            "Beitrag Gemeinschaft",
            "Private Überweisung Familie",
            "Einmalige private Hilfe",
            "Unklare regelmäßige Abbuchung",
        ),
    },
}

POSITIVE_CATEGORIES = {"income"}
COUNTERPARTIES = {
    "income": {"en": "Example Employer", "de": "Beispiel Arbeitgeber"},
    "investments": {"en": "Demo Securities Broker", "de": "Demo Wertpapierbroker"},
    "fees": {"en": "Example Bank", "de": "Beispielbank"},
    "taxes": {"en": "Revenue Authority", "de": "Finanzbehörde"},
    "savings": {"en": "Reserve Account", "de": "Rücklagenkonto"},
    "cash": {"en": "Central ATM", "de": "Geldautomat Zentrum"},
    "groceries": {"en": "Green Market", "de": "Grüner Markt"},
    "dining": {"en": "Riverside Kitchen", "de": "Küche am Fluss"},
    "transport": {"en": "City Mobility", "de": "Stadtmobilität"},
    "housing": {"en": "Example Property Services", "de": "Beispiel Hausverwaltung"},
    "utilities": {"en": "Home Services", "de": "Versorgung Zuhause"},
    "healthcare": {"en": "Community Health", "de": "Gesundheitszentrum"},
    "shopping": {"en": "Town Retail", "de": "Stadthandel"},
    "entertainment": {"en": "City Culture", "de": "Stadtkultur"},
    "travel": {"en": "Example Travel", "de": "Beispiel Reisen"},
    "insurance": {"en": "Demo Insurance", "de": "Demo Versicherung"},
    "education": {"en": "Learning Center", "de": "Bildungszentrum"},
    "other": {"en": "Reference Counterparty", "de": "Referenz Gegenpartei"},
}


def build_text_classification_challenge() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, languages in CURATED_TEXTS.items():
        for language, descriptions in languages.items():
            for index, description in enumerate(descriptions, start=1):
                difficulty = "easy" if index <= 2 else "medium" if index <= 5 else "hard"
                rows.append(
                    {
                        "id": f"{language}-{category}-{index:02d}",
                        "description": description,
                        "counterparty": (
                            COUNTERPARTIES[category][language] if index in {2, 5, 7} else ""
                        ),
                        "amount": (
                            125.0
                            if category in POSITIVE_CATEGORIES
                            or (category == "investments" and index in {1, 4})
                            else -42.5
                        ),
                        "expected_category": category,
                        "language": language,
                        "difficulty": difficulty,
                        "ambiguity": index >= 6,
                        "scenario_group": f"{language}-{category}-{index:02d}",
                        "notes": "",
                    }
                )
    challenge = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    validate_text_classification_challenge(challenge)
    return challenge


def validate_text_classification_challenge(challenge: pd.DataFrame) -> None:
    if list(challenge.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected challenge columns: {EXPECTED_COLUMNS}")
    if challenge.empty or challenge["id"].duplicated().any():
        raise ValueError("Challenge IDs must be non-empty and unique")
    expected_categories = {
        *(category.value for category in ExpenseCategory),
        *(category.value for category in TransactionCategory),
    }
    if set(challenge["expected_category"]) != expected_categories:
        raise ValueError("Challenge must cover every product category")
    if set(challenge["language"]) != {"en", "de"}:
        raise ValueError("Challenge must contain English and German cases")
    if set(challenge["difficulty"]) != {"easy", "medium", "hard"}:
        raise ValueError("Challenge must cover all difficulty levels")
    if challenge["description"].str.strip().eq("").any():
        raise ValueError("Challenge descriptions must not be blank")
    if challenge["amount"].eq(0).any():
        raise ValueError("Challenge amounts must not be zero")
    if challenge["counterparty"].str.strip().eq("").all():
        raise ValueError("Challenge must contain counterparty examples")
    if len(challenge) != len(expected_categories) * 2 * 7:
        raise ValueError("Challenge must contain seven cases per category and language")
    group_sizes = challenge.groupby(["expected_category", "language"]).size()
    if not group_sizes.eq(7).all():
        raise ValueError("Every category-language group must contain seven cases")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text_classification_challenge(
    destination: Path = DEFAULT_CHALLENGE_PATH,
    metadata_destination: Path = DEFAULT_METADATA_PATH,
) -> tuple[Path, Path]:
    challenge = build_text_classification_challenge()
    destination.parent.mkdir(parents=True, exist_ok=True)
    challenge.to_csv(destination, index=False)
    metadata = {
        "version": CHALLENGE_VERSION,
        "rows": len(challenge),
        "languages": sorted(challenge["language"].unique()),
        "categories": sorted(challenge["expected_category"].unique()),
        "sha256": _sha256(destination),
        "known_regression_cases_included": False,
        "supersedes": "text-classification-challenge-v1",
        "provenance": "manually_authored_development_challenge",
        "limitations": (
            "Created alongside the system; not an independent real-world production benchmark."
        ),
    }
    metadata_destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination, metadata_destination


def load_text_classification_challenge(
    source: Path = DEFAULT_CHALLENGE_PATH,
    metadata_source: Path | None = None,
) -> pd.DataFrame:
    metadata_path = metadata_source or source.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"Challenge metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != CHALLENGE_VERSION:
        raise ValueError("Challenge metadata version is incompatible")
    if metadata.get("sha256") != _sha256(source):
        raise ValueError("Challenge checksum does not match its metadata")
    challenge = pd.read_csv(source).fillna({"counterparty": "", "notes": ""})
    validate_text_classification_challenge(challenge)
    if metadata.get("rows") != len(challenge):
        raise ValueError("Challenge row count does not match its metadata")
    return challenge


def run() -> None:
    csv_path, metadata_path = write_text_classification_challenge()
    print(f"Text classification challenge ready: {csv_path} ({metadata_path})")


if __name__ == "__main__":
    run()
