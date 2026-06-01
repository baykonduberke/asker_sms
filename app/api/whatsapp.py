import xml.etree.ElementTree as ET
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])


def twiml_response(message: str = "") -> Response:
    """Cevabi Twilio'nun bekledigi TwiML (XML) formatina cevirir."""
    response = ET.Element("Response")
    if message:
        ET.SubElement(response, "Message").text = message
    xml = ET.tostring(response, encoding="unicode")
    return Response(content=xml, media_type="application/xml")


@router.post("/inbound")
async def whatsapp_inbound(request: Request) -> Response:
    """WhatsApp'tan gelen mesaja yapay zeka cevabi uretir.

    Akis: Twilio -> bu uc -> agent grafigi -> TwiML -> Twilio -> WhatsApp
    """

    raw_body = (await request.body()).decode("utf-8")

    fields = parse_qs(raw_body)  # orn: {"From": ["whatsapp:+90..."], "Body": ["merhaba"]}

    # parse_qs her alani LISTE verir; ilk degeri alip bosluklari temizliyoruz.
    sender = fields.get("From", [""])[0].strip()
    message = fields.get("Body", [""])[0].strip()

    # 2) Bos mesajda yapay zekayi calistirmadan bos cevap don.
    if not sender or not message:
        return twiml_response()

    # 3) Mesaji yapay zeka grafigine ver, cevabini al.
    #    graph.invoke yavas/bloklayici oldugu icin ayri thread'de calistiririz.
    graph = request.app.state.agent_graph
    result = await run_in_threadpool(
        graph.invoke,
        {"user_message": message},
        {"configurable": {"thread_id": sender}},  # ayni gonderen = ayni konusma hafizasi
    )
    # 4) Cevabi TwiML olarak Twilio'ya don.
    return twiml_response(result["agent_message"])