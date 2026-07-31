# Transaction category taxonomy

Taxonomy version: `transaction-categories-v1`

## Purpose

This document defines the category labels used for transaction categorization.
The definitions are intended to keep training data, model evaluation, API
responses, and user corrections consistent.

## Model scope

The learned model categorizes merchant-related expenses from transaction text.
The product service uses name/description and counterparty as text input plus
the signed amount for inflow/outflow routing. Detailed transaction types may be
stored as source metadata but are not classification features.

Text rules are direction-aware. An expense, fee, tax, savings, or cash phrase
on a positive cash flow and an income phrase on a negative cash flow are sent to
manual review instead of being forced into a contradictory category.

The following categories remain outside expense-model training and are handled
by a small text-rule baseline when high-signal phrases are present:

- salary, payroll, income, interest -> `income`
- dividend, broker, investment or security-purchase text -> `investments`
- bank/account fee text -> `fees`
- tax text -> `taxes`
- savings-account text -> `savings`
- ATM or cash-withdrawal text -> `cash`

Transfers are excluded from version 1 because their purpose is often ambiguous
without additional account or counterparty context.

## Labeling principles

- Assign exactly one category to each training example.
- Prefer the purpose of the purchase over the merchant's general business type.
- Apply the boundary rules below consistently.
- Use `other` only when no specific category can be supported by the available
  transaction text.
- Do not infer sensitive personal information from merchant names.
- Keep technical labels lowercase and stable across model versions.

## Categories

### `groceries`

Food, beverages, and everyday household consumables purchased primarily for use
at home.

Positive examples:

- `REWE MARKT 1842 BERLIN`
- `LIDL SAGT DANKE`
- `ALDI SUED FILIALE 927`

Boundary rules:

- Bakeries, cafés, restaurants, and food-delivery services are `dining`.
- Drugstores are `shopping` unless the description clearly identifies a
  pharmacy or medical purchase.

### `dining`

Prepared food or beverages purchased from restaurants, cafés, bars, bakeries,
canteens, or delivery services.

Positive examples:

- `CAFE CENTRAL BERLIN`
- `PIZZA DELIVERY SERVICE`
- `BACKEREI MUSTERMANN`

Boundary rules:

- Bakeries are classified as `dining` in version 1, even when the purchase may
  be consumed at home.
- Supermarket purchases remain `groceries`, including ready-to-eat products.

### `transport`

Local and long-distance ground transportation, fuel, parking, taxis, vehicle
charging, and public-transit services.

Positive examples:

- `DB VERTRIEB GMBH`
- `BVG TICKET APP`
- `SHELL STATION 428`

Boundary rules:

- Flights and expenses primarily associated with a trip are `travel`.
- Vehicle insurance is `insurance`; vehicle repairs are `other` in version 1.

### `housing`

Rent and payments to property managers or housing providers for the use of a
primary residence.

Positive examples:

- `MONTHLY APARTMENT RENT`
- `DEMO PROPERTY MANAGEMENT`
- `HOUSING COOPERATIVE PAYMENT`

Boundary rules:

- Electricity, gas, water, internet, and telecommunications are `utilities`.
- Furniture, appliances, and household goods are `shopping`.

### `utilities`

Recurring household services such as electricity, gas, water, internet,
telephone, and mobile service.

Positive examples:

- `ENERGY SUPPLIER MONTHLY BILL`
- `INTERNET PROVIDER GMBH`
- `MOBILE PHONE CONTRACT`

Boundary rules:

- Rent and property-management payments are `housing`.
- One-time purchases of phones, routers, or appliances are `shopping`.

### `healthcare`

Medical treatment, pharmacies, dental care, therapy, and other direct healthcare
services.

Positive examples:

- `APOTHEKE AM MARKT`
- `DENTAL PRACTICE DEMO`
- `PHYSIOTHERAPY CENTER`

Boundary rules:

- Health-insurance premiums are `insurance`.
- General drugstore purchases are `shopping` unless the transaction text clearly
  indicates a pharmacy or medical service.

### `shopping`

Retail purchases of clothing, electronics, household goods, personal-care
products, and other durable or discretionary goods.

Positive examples:

- `ELECTRONICS STORE ONLINE`
- `FASHION RETAIL BERLIN`
- `DM DROGERIE MARKT`

Boundary rules:

- Supermarket purchases are `groceries`.
- Tickets, games, and streaming subscriptions are `entertainment`.

### `entertainment`

Streaming, cinema, games, events, cultural activities, and recreational media.

Positive examples:

- `NETFLIX.COM`
- `CITY CINEMA TICKETS`
- `ONLINE GAME STORE`

Boundary rules:

- Restaurants and bars are `dining` even when visited for leisure.
- Hotels, airlines, and travel bookings are `travel`.

### `travel`

Flights, hotels, holiday accommodation, travel agencies, and expenses that are
clearly associated with a trip.

Positive examples:

- `DEMO AIRLINES BOOKING`
- `HOTEL RESERVATION BERLIN`
- `ONLINE TRAVEL AGENCY`

Boundary rules:

- Regular rail, taxi, fuel, and public-transit expenses are `transport` unless
  the transaction text explicitly identifies a travel package or trip.
- Restaurants remain `dining`, including restaurants visited while traveling.

### `insurance`

Premiums for health, liability, household, vehicle, travel, and other insurance
policies.

Positive examples:

- `LIABILITY INSURANCE PREMIUM`
- `DEMO HEALTH INSURANCE`
- `VEHICLE INSURANCE CONTRACT`

Boundary rules:

- Medical treatment is `healthcare`.
- A travel-agency booking is `travel`; only the insurance policy itself is
  `insurance`.

### `education`

Tuition, courses, professional training, educational platforms, and purchases
that are clearly identified as educational material.

Positive examples:

- `ONLINE COURSE PLATFORM`
- `UNIVERSITY TUITION PAYMENT`
- `LANGUAGE SCHOOL BERLIN`

Boundary rules:

- General bookstore purchases are `shopping` unless the transaction description
  clearly identifies educational material.
- Entertainment subscriptions are `entertainment`.

### `other`

Merchant-related expenses that cannot be assigned reliably to a more specific
version 1 category from the available text.

Positive examples:

- `MISCELLANEOUS SERVICE PAYMENT`
- `UNKNOWN MERCHANT 48291`
- `GENERAL PAYMENT REFERENCE`

Boundary rules:

- `other` must not be used merely because a merchant name is unfamiliar; labelers
  should first use the available description and documented rules.
- Ambiguous examples should be flagged for review and excluded from high-quality
  evaluation sets when no defensible ground-truth label exists.

## Categories outside model scope

The following labels may appear in the product but are assigned by the text-rule
baseline rather than predicted by the version 1 expense model:

- `income`
- `investments`
- `fees`
- `taxes`
- `savings`
- `cash`

## Versioning

Changing a label name, merging categories, splitting a category, or changing a
boundary rule requires a new taxonomy version. Training datasets and model
artifacts must record the taxonomy version they use.
