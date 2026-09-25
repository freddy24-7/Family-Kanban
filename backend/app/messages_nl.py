"""Dutch user-facing email texts (UI strings live in the frontend)."""


def verify_email(link: str) -> tuple[str, str]:
    return (
        "Bevestig je e-mailadres",
        f"Welkom bij Gezinsbord!\n\nBevestig je e-mailadres via deze link:\n{link}\n",
    )


def reset_password(link: str) -> tuple[str, str]:
    return (
        "Wachtwoord opnieuw instellen",
        "Je hebt gevraagd je wachtwoord opnieuw in te stellen.\n\n"
        f"Gebruik deze link (1 uur geldig):\n{link}\n\n"
        "Heb je dit niet aangevraagd? Dan kun je deze e-mail negeren.\n",
    )


def household_invite(household_name: str, inviter_name: str, link: str) -> tuple[str, str]:
    return (
        f"Uitnodiging voor {household_name}",
        f"{inviter_name} nodigt je uit voor het gezinsbord '{household_name}'.\n\n"
        f"Accepteer de uitnodiging via deze link:\n{link}\n",
    )
