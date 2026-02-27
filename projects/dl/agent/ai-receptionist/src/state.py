from typing import TypedDict, Optional

class ReceptionistState(TypedDict):

    user_input: str

    intent: Optional[str]

    name: Optional[str]

    date_preference: Optional[str]

    phone: Optional[str]

    message: Optional[str]

    response: Optional[str]