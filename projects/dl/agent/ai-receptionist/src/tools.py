import random

FAQ = {
    "horaires": "Nous sommes ouverts du lundi au vendredi de 9h à 18h et le samedi de 9h à 12h.",
    "adresse": "Le cabinet est situé au 12 rue des Lilas à Paris.",
    "tarifs": "La consultation coûte 30 euros et une consultation urgente coûte 50 euros."
}


def faq_lookup(question: str):

    question = question.lower()

    if "horaire" in question:
        return FAQ["horaires"]

    if "adresse" in question:
        return FAQ["adresse"]

    if "tarif" in question or "prix" in question:
        return FAQ["tarifs"]

    return "Je peux vous renseigner sur les horaires, l'adresse ou les tarifs."


def book_appointment(name, date_preference, reason=None):

    appointment_id = random.randint(1000, 9999)

    return {
        "appointment_id": appointment_id,
        "slot": date_preference
    }


def send_message(name, phone, message):

    ticket_id = random.randint(10000, 99999)

    return {
        "ticket_id": ticket_id
    }