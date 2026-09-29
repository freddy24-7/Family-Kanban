# Labelling guidelines

The single source of truth for what each label means. The same definitions are used
by the Gemini ticket generator (weak labels), shown to planners when they confirm or
correct a prediction, and used when reviewing data. If humans and the generator label
by different rules, the model learns a mixture and the metrics become meaningless.

Change these rules only deliberately: a changed definition changes the target the
model learns, which makes metrics before and after the change incomparable. Record
changes at the bottom with a date.

## Categories

Decide by **purpose / who it is for** first, then by activity. "Buying" alone never
decides the category.

| Key | Dutch label | Covers | Examples |
|---|---|---|---|
| `chores` | Huishouden | Recurring work keeping the household running: cleaning, laundry, dishes, tidying, trash, cooking, routine pet care (feeding, walking, litter box) | *Was ophangen*, *Hond uitlaten*, *Koelkast schoonmaken* |
| `groceries` | Boodschappen | Buying routine consumables: food, drinks, toiletries, cleaning products, pet food | *Melk en brood halen*, *Kattenbakvulling kopen* |
| `kids` | Kinderen | Care, school and activities of the household's children | *Gymtas Sem inpakken*, *Ouderavond school*, *Lotte naar voetbal brengen*, *Nieuwe schoenen voor Noor* |
| `home_maintenance` | Klussen & onderhoud | Repairs, upkeep and improvements of house, garden, car and bikes; buying durable items for that | *Kraan repareren*, *Gras maaien*, *Dakgoot schoonmaken*, *APK auto*, *Fietsband plakken* |
| `finance` | Financiën | Money and paperwork with a financial side: bills, taxes, insurance, subscriptions, transfers, budgeting | *Belastingaangifte doen*, *Energiecontract vergelijken*, *Zorgverzekering betalen* |
| `social` | Sociaal | People outside the household: birthdays, gifts, visits, dinners, cards, family events | *Cadeau voor oma kopen*, *Etentje met buren plannen* |
| `other` | Overig | None of the above: adults' personal appointments, non-financial admin, hobbies | *Paspoort verlengen*, *Tandarts (volwassene)* |

### Boundary rules (the cases people disagree on)

- Pet **food** is `groceries`; pet **care** (feeding, walking, vet visit) is `chores`.
- Garden work (mowing, pruning, weeding) is `home_maintenance`, even when routine.
- Anything whose main subject is a child of the household is `kids`, even if it involves
  buying (*schoenen voor Noor*) or an appointment (*tandarts Sem*).
- A gift or event for someone outside the household is `social`, even if a child attends.
- Cooking dinner is `chores`; shopping for its ingredients is `groceries`.

## Effort

Total **person-time** needed, not calendar time until it's done.

| Key | Meaning |
|---|---|
| `S` | up to 30 minutes |
| `M` | 30 minutes to 2 hours |
| `L` | more than 2 hours, or several sessions |

Effort is ordinal: S < M < L. Mistaking L for S is worse than mistaking L for M.

## Change log

- 2026-09-29: initial version.
