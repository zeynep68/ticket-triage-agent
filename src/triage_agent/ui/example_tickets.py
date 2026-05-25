"""Curated example tickets for the live-triage demo.

These examples cover the five terminal-action paths and the multi-turn loop:
- direct FORWARD (routine)
- FORWARD with clarifications (partial info)
- CLAIM (concrete damage report)
- ESCALATE (explicit legal trigger)
- CLARIFY (too vague to route)
- FAQ (generic how-to)

The list contains German and English versions of the same eight scenarios so
the language-agnostic behaviour can be inspected side by side.
"""

EXAMPLE_TICKETS: dict[str, str] = {
    # German
    "Routine Rechnungs-Anfrage (FORWARD)": (
        "Bitte resenden Sie mir meine Police-Rechnung für April 2026. "
        "Vertragsnummer HV-78342."
    ),
    "Diebstahl-Schaden (CLAIM)": (
        "Sehr geehrte Damen und Herren,\n\n"
        "mein Auto wurde gestern Nacht aus der Tiefgarage gestohlen. "
        "Vertragsnummer KFZ-12345. Ich brauche eine Schadensmeldung und "
        "wissen, welche Unterlagen ich beilegen muss.\n\n"
        "Mit freundlichen Grüßen"
    ),
    "Anwalts-Erwähnung (ESCALATE)": (
        "Sehr geehrtes Support-Team,\n\n"
        "ich werde meinen Anwalt einschalten. Seit sechs Wochen warte ich "
        "auf eine Antwort zu meiner Schadensregulierung vom 15. März. "
        "Vorgangsnummer SCH-2026-0451. Ich erwäge zudem die Kündigung "
        "meines Vertrags.\n\n"
        "Mit freundlichen Grüßen"
    ),
    "Vages Ticket (CLARIFY)": (
        "Hilfe, das System funktioniert nicht!"
    ),
    "Passwort-Reset (FAQ)": (
        "Hallo,\n\nwie kann ich mein Passwort im Kundenportal zurücksetzen? "
        "Vielen Dank."
    ),
    "Login-Problem (FORWARD + Rückfragen)": (
        "Sehr geehrtes Support-Team,\n\n"
        "der Login in mein Kundenkonto funktioniert seit gestern nicht mehr. "
        "Ich erhalte keine konkrete Fehlermeldung, die Seite lädt einfach "
        "nicht.\n\nMit freundlichen Grüßen"
    ),
    "Wasserschaden (CLAIM)": (
        "Guten Tag,\n\n"
        "wir haben einen Wasserschaden in unserer Wohnung — die "
        "Waschmaschine ist heute Nacht ausgelaufen, mehrere Räume sind "
        "betroffen. Vertragsnummer HV-99201. Wie gehe ich am besten vor?\n\n"
        "Vielen Dank"
    ),
    "Business-Impact (Implizit High)": (
        "Sehr geehrtes Support-Team,\n\n"
        "wir haben seit drei Tagen erhebliche Verzögerungen bei der "
        "Schadensregulierung unserer Firmen-KFZ-Verträge. Das beeinträchtigt "
        "unsere Buchhaltung und verzögert die Abrechnung gegenüber unseren "
        "Kunden. Wir bitten um eine zeitnahe Klärung.\n\n"
        "Mit freundlichen Grüßen"
    ),
    # English equivalents
    "Routine invoice request (FORWARD)": (
        "Please resend my insurance policy invoice for April 2026. "
        "Contract number HV-78342."
    ),
    "Vehicle theft claim (CLAIM)": (
        "Dear Sir or Madam,\n\n"
        "my car was stolen last night from the underground garage. "
        "Contract number KFZ-12345. I need to file a damage report and "
        "would like to know which documents I should include.\n\n"
        "Kind regards"
    ),
    "Lawyer mention (ESCALATE)": (
        "Dear Support Team,\n\n"
        "I will be engaging my lawyer. I have been waiting six weeks for "
        "a response regarding my claim settlement from March 15. Case "
        "number SCH-2026-0451. I am also considering canceling my contract.\n\n"
        "Kind regards"
    ),
    "Vague ticket (CLARIFY)": (
        "Help, the system isn't working!"
    ),
    "Password reset (FAQ)": (
        "Hello,\n\nhow can I reset my password in the customer portal? "
        "Thank you."
    ),
    "Login problem (FORWARD + questions)": (
        "Dear Support Team,\n\n"
        "logging into my customer account has not been working since "
        "yesterday. I do not get a specific error message, the page simply "
        "fails to load.\n\nKind regards"
    ),
    "Water damage (CLAIM)": (
        "Good day,\n\n"
        "we have water damage in our apartment — the washing machine "
        "leaked overnight, several rooms are affected. Contract number "
        "HV-99201. What is the best way to proceed?\n\n"
        "Thank you"
    ),
    "Business impact (implicit high)": (
        "Dear Support Team,\n\n"
        "for three days now we have been experiencing significant delays "
        "in the claim settlement of our corporate vehicle contracts. This "
        "affects our accounting and delays invoicing to our customers. "
        "We ask for prompt clarification.\n\n"
        "Kind regards"
    ),
}
