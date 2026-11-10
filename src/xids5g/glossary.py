"""Plain-English names for the 5G-NIDD columns.

Objective 5 requires output a non-expert decision-maker can act on, so every
feature the model may cite needs a readable name, a unit and a one-line
explanation of what it measures. Anything not listed here falls back to the raw
column name, which keeps the output honest rather than inventing a description.
"""

from __future__ import annotations

#: column -> (readable name, unit, what it measures)
GLOSSARY: dict[str, tuple[str, str, str]] = {
    "Dur": ("flow duration", "s", "how long the conversation lasted"),
    "RunTime": ("active time", "s", "time the flow was actually transmitting"),
    "Mean": ("mean inter-packet gap", "s", "average time between packets"),
    "Sum": ("total active time", "s", "summed transmission time"),
    "Min": ("shortest packet gap", "s", "smallest time between two packets"),
    "Max": ("longest packet gap", "s", "largest time between two packets"),
    "Proto": ("protocol", "", "transport protocol (TCP, UDP, ICMP, ...)"),
    "sTos": ("source type-of-service", "", "IP priority marking set by the sender"),
    "dTos": ("destination type-of-service", "", "IP priority marking seen in replies"),
    "sDSb": ("source DSCP class", "", "quality-of-service class set by the sender"),
    "dDSb": ("destination DSCP class", "", "quality-of-service class in replies"),
    "sTtl": ("source time-to-live", "hops", "remaining hop budget on packets from the sender"),
    "dTtl": ("destination time-to-live", "hops", "remaining hop budget on replies"),
    "sHops": ("hop count from the source", "hops", "estimated router hops travelled from the sender"),
    "dHops": ("hop count from the destination", "hops", "estimated router hops travelled from the peer"),
    "Cause": ("record type", "", "why this record was written (flow start, status, shutdown)"),
    "TotPkts": ("total packet count", "packets", "packets exchanged in both directions"),
    "SrcPkts": ("source packet count", "packets", "packets sent by the initiator"),
    "DstPkts": ("destination packet count", "packets", "packets sent back by the peer"),
    "TotBytes": ("total byte count", "bytes", "volume exchanged in both directions"),
    "SrcBytes": ("source byte count", "bytes", "volume sent by the initiator"),
    "DstBytes": ("destination byte count", "bytes", "volume sent back by the peer"),
    "sMeanPktSz": ("mean packet size (source)", "bytes", "average size of packets sent"),
    "dMeanPktSz": ("mean packet size (destination)", "bytes", "average size of packets received"),
    "Load": ("throughput", "bits/s", "overall data rate of the flow"),
    "SrcLoad": ("throughput from source", "bits/s", "data rate from the initiator"),
    "DstLoad": ("throughput from destination", "bits/s", "data rate from the peer"),
    "Loss": ("lost-packet count", "packets", "packets that had to be retransmitted"),
    "SrcLoss": ("source lost-packet count", "packets", "retransmissions by the initiator"),
    "DstLoss": ("destination lost-packet count", "packets", "retransmissions by the peer"),
    "pLoss": ("packet loss", "%", "share of packets lost"),
    "SrcGap": ("source gap count", "", "sequence-number gaps seen from the initiator"),
    "DstGap": ("destination gap count", "", "sequence-number gaps seen from the peer"),
    "Rate": ("packet rate", "packets/s", "how fast packets were sent overall"),
    "SrcRate": ("packet rate from source", "packets/s", "sending rate of the initiator"),
    "DstRate": ("packet rate from destination", "packets/s", "sending rate of the peer"),
    "State": ("connection state", "", "how the connection ended (established, reset, request only, ...)"),
    "SrcWin": ("TCP window (source)", "bytes", "receive buffer advertised by the initiator"),
    "DstWin": ("TCP window (destination)", "bytes", "receive buffer advertised by the peer"),
    "SrcTCPBase": ("TCP base sequence (source)", "", "initial TCP sequence number from the initiator"),
    "DstTCPBase": ("TCP base sequence (destination)", "", "initial TCP sequence number from the peer"),
    "TcpRtt": ("TCP round-trip time", "s", "measured handshake round-trip delay"),
    "SynAck": ("SYN-to-ACK time", "s", "delay between connection request and acknowledgement"),
    "AckDat": ("ACK-to-data time", "s", "delay between acknowledgement and first data"),
}

#: Short operator-facing description of each attack class.
ATTACK_DESCRIPTIONS: dict[str, str] = {
    "Benign": "Normal traffic. No action required.",
    "UDPFlood": "A denial-of-service flood of UDP packets aimed at exhausting "
                "bandwidth or server capacity.",
    "HTTPFlood": "A denial-of-service flood of HTTP requests designed to "
                 "overwhelm a web service.",
    "SlowrateDoS": "A slow-rate denial-of-service attack that holds connections "
                   "open to exhaust the server's connection pool.",
    "SYNFlood": "A TCP SYN flood: half-open connections opened faster than the "
                "target can close them.",
    "ICMPFlood": "A flood of ICMP (ping) traffic used to saturate the link.",
    "SYNScan": "Reconnaissance. TCP SYN probes used to map which ports are open.",
    "TCPConnectScan": "Reconnaissance. Full TCP connections opened to identify "
                      "listening services.",
    "UDPScan": "Reconnaissance. UDP probes used to discover open UDP services.",
}

#: Recommended first response, phrased for a non-expert decision-maker.
RECOMMENDED_ACTION: dict[str, str] = {
    "Benign": "None.",
    "UDPFlood": "Rate-limit or block the source at the edge; verify the target "
                "service is still reachable for legitimate users.",
    "HTTPFlood": "Apply request rate-limiting for the source; check web server "
                 "load and enable challenge/CAPTCHA if available.",
    "SlowrateDoS": "Reduce the connection timeout on the target service and cap "
                   "concurrent connections per source address.",
    "SYNFlood": "Enable SYN cookies on the target host and rate-limit new "
                "connections from the source.",
    "ICMPFlood": "Rate-limit ICMP at the gateway; confirm no diagnostic tooling "
                 "is responsible before blocking.",
    "SYNScan": "Log and monitor the source. Scanning usually precedes an "
               "attack; review what it found exposed.",
    "TCPConnectScan": "Log and monitor the source; audit which services it "
                      "successfully connected to.",
    "UDPScan": "Log and monitor the source; review exposed UDP services.",
}


def describe(feature: str) -> tuple[str, str, str]:
    """Readable name, unit and meaning for a feature, with a safe fallback."""
    base = feature.replace("__not_observed", "")
    name, unit, what = GLOSSARY.get(
        base, (base, "", "a measured property of the flow"))
    if feature.endswith("__not_observed"):
        return (f"{name} was not observed", "",
                f"there was no {name} to measure on this flow")
    return name, unit, what


def format_value(feature: str, value: float, decode=None) -> str:
    """Render a feature value the way an operator would expect to see it."""
    if feature.endswith("__not_observed"):
        return "yes" if value >= 0.5 else "no"
    if decode is not None:
        return str(decode)
    _, unit, _ = describe(feature)
    if unit in ("packets", "bytes", "hops"):
        # "1 hops" reads badly on a card an operator is meant to trust.
        word = unit[:-1] if (abs(value - 1) < 1e-9 and unit.endswith("s")) else unit
        return f"{value:,.0f} {word}".strip()
    if unit == "%":
        return f"{value:.2f}%"
    if unit in ("s",):
        return f"{value:.4g} s"
    if unit == "bits/s":
        return f"{value:,.0f} bits/s"
    if unit == "packets/s":
        return f"{value:,.1f} packets/s"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.4g}"
