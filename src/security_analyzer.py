"""
PhishGuard - Cybersecurity Analysis Engine
Handles RFC 822 / 2822 email parsing (.eml files & raw text), header analysis,
IOC extraction (IPs, URLs, domains), sender mismatch detection, SPF/DKIM/DMARC
header extraction, and composite Cyber Threat Risk Scoring (0-100).
SAFETY GUARANTEE: Performs purely passive, static syntactic inspection.
Never performs active network requests to extracted URLs or executes attachments.
"""

import re
import email
from email import policy
from email.parser import BytesParser, Parser
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

# Suspicious file extensions commonly delivered in malicious email attachments
HIGH_RISK_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".jse", ".wsf",
    ".hta", ".cpl", ".iso", ".img", ".vhd", ".docm", ".xlsm", ".pptm",
    ".dll", ".pif", ".application", ".gadget", ".jar"
}

# Suspicious TLDs frequently abused by low-cost threat campaigns
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".tk", ".pw", ".buzz", ".ru", ".cn", ".club", ".work",
    ".support", ".live", ".stream", ".icu", ".monster", ".gq", ".ga", ".ml", ".cf"
}

# Major protected brands frequently impersonated
KNOWN_BRANDS = [
    "paypal", "microsoft", "office365", "google", "apple", "amazon",
    "netflix", "chase", "bank of america", "wells fargo", "citibank",
    "dhl", "fedex", "ups", "irs", "dropbox", "linkedin"
]

# Official canonical domains for monitored brands
OFFICIAL_BRAND_DOMAINS = {
    "paypal": ["paypal.com"],
    "microsoft": ["microsoft.com", "office.com", "live.com", "outlook.com"],
    "google": ["google.com", "gmail.com"],
    "apple": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "aws.com"],
    "netflix": ["netflix.com"],
    "chase": ["chase.com", "jpmorgan.com"],
    "bank of america": ["bankofamerica.com", "bofa.com"],
    "wells fargo": ["wellsfargo.com"],
    "dhl": ["dhl.com"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "irs": ["irs.gov"],
    "linkedin": ["linkedin.com"]
}

class SecurityAnalyzer:
    """
    Forensic security analyzer for email messages.
    Extracts Indicators of Compromise (IOCs), parses headers, detects spoofing,
    and calculates an explainable Threat Risk Score.
    """
    
    def __init__(self):
        self.url_regex = re.compile(r'https?://[^\s<>"\'\)]+|www\.[^\s<>"\'\)]+', re.IGNORECASE)
        self.ip_regex = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    def parse_eml_content(self, eml_bytes_or_str) -> Dict[str, Any]:
        """
        Parses raw .eml file content (bytes or string) into headers, body, and attachments.
        Handles indented or pasted headers cleanly.
        """
        import textwrap
        if isinstance(eml_bytes_or_str, str):
            clean_str = textwrap.dedent(eml_bytes_or_str).lstrip()
            msg = Parser(policy=policy.default).parsestr(clean_str)
        else:
            msg = BytesParser(policy=policy.default).parsebytes(eml_bytes_or_str)
        
        headers = {}
        for k, v in msg.items():
            headers[k] = str(v)
            
        # Extract body text and HTML
        body_text = ""
        body_html = ""
        attachments = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if "attachment" in content_disposition.lower() or part.get_filename():
                    filename = part.get_filename() or "unnamed_attachment"
                    size_bytes = len(part.get_payload(decode=True) or b"")
                    attachments.append({
                        "filename": filename,
                        "content_type": content_type,
                        "size_bytes": size_bytes,
                        "is_risky": any(filename.lower().endswith(ext) for ext in HIGH_RISK_EXTENSIONS)
                    })
                elif content_type == "text/plain" and not body_text:
                    try:
                        body_text = part.get_content()
                    except Exception:
                        payload = part.get_payload(decode=True)
                        body_text = payload.decode(errors="replace") if payload else ""
                elif content_type == "text/html" and not body_html:
                    try:
                        body_html = part.get_content()
                    except Exception:
                        payload = part.get_payload(decode=True)
                        body_html = payload.decode(errors="replace") if payload else ""
        else:
            content_type = msg.get_content_type()
            try:
                content = msg.get_content()
            except Exception:
                payload = msg.get_payload(decode=True)
                content = payload.decode(errors="replace") if payload else ""
                
            if content_type == "text/html":
                body_html = content
            else:
                body_text = content
                
        # Primary body for text analysis
        full_text = body_text if body_text else body_html
        
        return {
            "headers": headers,
            "subject": headers.get("Subject", "(No Subject)"),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "return_path": headers.get("Return-Path", ""),
            "reply_to": headers.get("Reply-To", ""),
            "date": headers.get("Date", ""),
            "message_id": headers.get("Message-ID", ""),
            "body_text": full_text,
            "body_html": body_html,
            "attachments": attachments
        }
    
    def extract_iocs(self, text: str) -> Dict[str, Any]:
        """
        Extracts Indicators of Compromise (IOCs): URLs, domains, and IP addresses.
        Defangs URLs and IPs for safe presentation.
        """
        raw_urls = self.url_regex.findall(text)
        # Deduplicate while preserving order
        seen_urls = set()
        clean_urls = []
        for u in raw_urls:
            u_clean = u.rstrip(".,;)>'\"")
            if u_clean not in seen_urls:
                seen_urls.add(u_clean)
                clean_urls.append(u_clean)
        
        extracted_domains = []
        suspicious_urls = []
        ip_based_urls = []
        
        for url in clean_urls:
            parsed = urlparse(url if "://" in url else "http://" + url)
            domain = parsed.netloc.lower().split(":")[0]
            if domain:
                extracted_domains.append(domain)
                # Check for IP in domain
                if self.ip_regex.fullmatch(domain):
                    ip_based_urls.append(url)
                    suspicious_urls.append({
                        "url": url,
                        "defanged": self.defang_url(url),
                        "reason": "Direct IP-address host (circumvents DNS reputation checks)"
                    })
                # Check for suspicious TLD
                elif any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
                    suspicious_urls.append({
                        "url": url,
                        "defanged": self.defang_url(url),
                        "reason": f"High-risk top-level domain abuse: {domain}"
                    })
                # Check for embedded credentials or '@' sign
                elif "@" in url:
                    suspicious_urls.append({
                        "url": url,
                        "defanged": self.defang_url(url),
                        "reason": "URL contains '@' sign (redirects to domain following @)"
                    })
        
        # Raw IPs found in text
        raw_ips = self.ip_regex.findall(text)
        unique_ips = list(dict.fromkeys(raw_ips))
        defanged_ips = [self.defang_ip(ip) for ip in unique_ips]
        
        return {
            "total_urls": len(clean_urls),
            "urls": clean_urls,
            "defanged_urls": [self.defang_url(u) for u in clean_urls],
            "domains": list(dict.fromkeys(extracted_domains)),
            "suspicious_urls": suspicious_urls,
            "ip_based_urls": ip_based_urls,
            "ip_addresses": unique_ips,
            "defanged_ips": defanged_ips
        }
    
    @staticmethod
    def defang_url(url: str) -> str:
        """Defangs a URL for safe UI presentation (e.g. hxxp://evil[.]com)."""
        return url.replace("http://", "hxxp://").replace("https://", "hxxps://").replace(".", "[.]")
    
    @staticmethod
    def defang_ip(ip: str) -> str:
        """Defangs an IP address for safe UI presentation (e.g. 192[.]168[.]1[.]1)."""
        return ip.replace(".", "[.]")
    
    def check_sender_mismatch(self, from_header: str, return_path: str = "", reply_to: str = "") -> Dict[str, Any]:
        """
        Analyzes From display name vs actual mailbox address, and From vs Return-Path / Reply-To.
        Detects brand impersonation and lookalike typosquatting domains.
        """
        mismatches = []
        is_suspicious = False
        
        # Extract display name and email address from "From: Display Name <user@domain.com>"
        match = re.search(r'^(.*?)\s*<([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})>$', from_header.strip())
        display_name = ""
        from_email = from_header.strip()
        from_domain = ""
        
        if match:
            display_name = match.group(1).replace('"', '').replace("'", "").strip()
            from_email = match.group(2).strip()
            from_domain = from_email.split("@")[-1].lower()
        elif "@" in from_header:
            email_match = self.email_regex.search(from_header)
            if email_match:
                from_email = email_match.group(0)
                from_domain = from_email.split("@")[-1].lower()
                
        # 1. Brand Impersonation & Typosquatting Analysis
        for brand, valid_domains in OFFICIAL_BRAND_DOMAINS.items():
            claimed_in_display = display_name and brand in display_name.lower()
            claimed_in_domain = brand in from_domain
            is_valid_official = any(from_domain == vd or from_domain.endswith("." + vd) for vd in valid_domains)
            
            if (claimed_in_display or claimed_in_domain) and not is_valid_official:
                is_suspicious = True
                mismatches.append({
                    "type": "Brand Impersonation / Lookalike Domain",
                    "severity": "HIGH",
                    "description": f"Claims affinity with '{brand.title()}', but sender domain '{from_domain}' is not an official domain (expected: {', '.join(valid_domains)})."
                })
                break
                    
        # 2. Return-Path Mismatch
        if return_path and from_domain:
            rp_match = self.email_regex.search(return_path)
            if rp_match:
                rp_email = rp_match.group(0)
                rp_domain = rp_email.split("@")[-1].lower()
                if rp_domain and rp_domain != from_domain:
                    # Common with bulk senders, but noteworthy
                    mismatches.append({
                        "type": "Return-Path Divergence",
                        "severity": "MEDIUM",
                        "description": f"Envelope Return-Path ({rp_domain}) does not match From header domain ({from_domain})."
                    })
                    
        # 3. Reply-To Mismatch
        if reply_to and from_domain:
            rt_match = self.email_regex.search(reply_to)
            if rt_match:
                rt_email = rt_match.group(0)
                rt_domain = rt_email.split("@")[-1].lower()
                if rt_domain and rt_domain != from_domain:
                    is_suspicious = True
                    mismatches.append({
                        "type": "Reply-To Redirection",
                        "severity": "HIGH",
                        "description": f"Replies are rerouted to external domain '{rt_domain}', diverging from sender '{from_domain}'."
                    })
                    
        return {
            "display_name": display_name,
            "from_email": from_email,
            "from_domain": from_domain,
            "has_mismatch": len(mismatches) > 0,
            "is_suspicious": is_suspicious,
            "mismatches": mismatches
        }
    
    def parse_auth_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Extracts reported SPF, DKIM, and DMARC results from email headers.
        Note: PhishGuard inspects embedded header tokens without claiming independent server verification.
        """
        auth_results = headers.get("Authentication-Results", "")
        received_spf = headers.get("Received-SPF", "")
        dkim_sig = headers.get("DKIM-Signature", "")
        
        spf_status = "NOT FOUND"
        dkim_status = "NOT FOUND"
        dmarc_status = "NOT FOUND"
        details = []
        
        raw_combined = f"{auth_results} {received_spf}".lower()
        
        # Check SPF
        if "spf=pass" in raw_combined or "pass" in received_spf.lower():
            spf_status = "PASS"
        elif "spf=fail" in raw_combined or "fail" in received_spf.lower():
            spf_status = "FAIL"
            details.append("Header records report SPF authentication failure.")
        elif "spf=softfail" in raw_combined:
            spf_status = "SOFTFAIL"
            details.append("Header records report SPF softfail (unauthorized sending server).")
        elif "spf=neutral" in raw_combined:
            spf_status = "NEUTRAL"
            
        # Check DKIM
        if "dkim=pass" in raw_combined:
            dkim_status = "PASS"
        elif "dkim=fail" in raw_combined:
            dkim_status = "FAIL"
            details.append("DKIM cryptographic signature verification failed.")
        elif dkim_sig:
            dkim_status = "PRESENT (Unverified)"
            
        # Check DMARC
        if "dmarc=pass" in raw_combined:
            dmarc_status = "PASS"
        elif "dmarc=fail" in raw_combined:
            dmarc_status = "FAIL"
            details.append("DMARC alignment policy failed.")
            
        return {
            "spf_status": spf_status,
            "dkim_status": dkim_status,
            "dmarc_status": dmarc_status,
            "raw_auth_results": auth_results or received_spf or "No Authentication-Results header located in source.",
            "details": details,
            "has_failures": any(s in ["FAIL", "SOFTFAIL"] for s in [spf_status, dkim_status, dmarc_status])
        }
    
    def calculate_risk_score(
        self,
        ml_probability: float,
        cyber_features: Dict[str, Any],
        iocs: Dict[str, Any],
        sender_analysis: Dict[str, Any],
        auth_analysis: Dict[str, Any],
        attachments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates a transparent cybersecurity Threat Risk Score (0-100)
        with an itemized breakdown of contributing factors.
        """
        factors = []
        score = 0.0
        
        # 1. Machine Learning Component (Max 40 points)
        ml_points = round(ml_probability * 40.0, 1)
        score += ml_points
        factors.append({
            "category": "Machine Learning Prediction",
            "points": ml_points,
            "max": 40,
            "description": f"Statistical NLP model phishing probability: {ml_probability * 100:.1f}%"
        })
        
        # 2. Sender Identity & Spoofing (Max 20 points)
        sender_points = 0.0
        if sender_analysis.get("is_suspicious"):
            sender_points += 15.0
            factors.append({
                "category": "Sender Identity Spoofing",
                "points": 15.0,
                "max": 15,
                "description": "High-confidence display name or brand spoofing detected."
            })
        elif sender_analysis.get("has_mismatch"):
            sender_points += 8.0
            factors.append({
                "category": "Sender Header Discrepancy",
                "points": 8.0,
                "max": 15,
                "description": "Discrepancy between From domain and Return-Path / Reply-To."
            })
            
        if auth_analysis.get("has_failures"):
            sender_points += 5.0
            factors.append({
                "category": "Email Authentication Failure",
                "points": 5.0,
                "max": 5,
                "description": f"Header verification failure (SPF: {auth_analysis['spf_status']}, DKIM: {auth_analysis['dkim_status']})."
            })
        score += min(sender_points, 20.0)
        
        # 3. Malicious URL & Network IOCs (Max 20 points)
        url_points = 0.0
        if len(iocs.get("ip_based_urls", [])) > 0:
            url_points += 12.0
            factors.append({
                "category": "IP-Based URL Threat",
                "points": 12.0,
                "max": 12,
                "description": f"{len(iocs['ip_based_urls'])} URL(s) bypass DNS by targeting direct raw IP hosts."
            })
        if len(iocs.get("suspicious_urls", [])) > 0:
            url_points += 8.0
            factors.append({
                "category": "Suspicious TLD / Obfuscation",
                "points": 8.0,
                "max": 8,
                "description": f"{len(iocs['suspicious_urls'])} URL(s) exhibit risky TLDs or credential-masking syntax."
            })
        score += min(url_points, 20.0)
        
        # 4. Social Engineering & Urgency Cues (Max 10 points)
        social_points = 0.0
        urgency_cnt = cyber_features.get("urgency_count", 0)
        cred_cnt = cyber_features.get("credential_count", 0)
        fin_cnt = cyber_features.get("financial_count", 0)
        
        if cred_cnt > 0:
            social_points += 5.0
            factors.append({
                "category": "Credential Harvesting Cues",
                "points": 5.0,
                "max": 5,
                "description": f"Detected {cred_cnt} credential-requesting expressions (passwords, verify account, login)."
            })
        if urgency_cnt > 0:
            social_points += 3.0
            factors.append({
                "category": "Psychological Coercion / Urgency",
                "points": 3.0,
                "max": 3,
                "description": f"Detected {urgency_cnt} psychological urgency cues (immediate, 24 hours, suspended)."
            })
        if fin_cnt > 0:
            social_points += 2.0
            factors.append({
                "category": "Financial Lure Cues",
                "points": 2.0,
                "max": 2,
                "description": f"Detected {fin_cnt} financial transaction or billing lures."
            })
        score += min(social_points, 10.0)
        
        # 5. Attachment & Payload Risk (Max 10 points)
        payload_points = 0.0
        risky_att = [a for a in attachments if a.get("is_risky")]
        if len(risky_att) > 0:
            payload_points += 10.0
            factors.append({
                "category": "Executable / Risky Attachment",
                "points": 10.0,
                "max": 10,
                "description": f"High-risk attachment detected: {', '.join(a['filename'] for a in risky_att)}"
            })
        elif len(attachments) > 0:
            payload_points += 2.0
            factors.append({
                "category": "Attachment Presence",
                "points": 2.0,
                "max": 2,
                "description": f"{len(attachments)} non-executable attachment(s) attached."
            })
        score += min(payload_points, 10.0)
        
        # Final normalized score bounded strictly in [0, 100]
        final_score = int(round(min(max(score, 0.0), 100.0)))
        
        # Risk classification level
        if final_score >= 70:
            risk_level = "HIGH CRITICAL RISK"
            verdict = "Phishing Threat"
            badge_color = "#f85149" # Red
        elif final_score >= 35:
            risk_level = "SUSPICIOUS / ELEVATED"
            verdict = "Suspicious"
            badge_color = "#d29922" # Amber
        else:
            risk_level = "LOW RISK"
            verdict = "Legitimate"
            badge_color = "#3fb950" # Green
            
        return {
            "score": final_score,
            "risk_level": risk_level,
            "verdict": verdict,
            "badge_color": badge_color,
            "factors": factors
        }
    
    def find_suspicious_snippets(self, text: str) -> List[Dict[str, str]]:
        """
        Locates exact keyword matches in the email text for contextual visual highlighting.
        """
        snippets = []
        categories = [
            ("Urgency", [r'\burgent\b', r'\bimmediate(?:ly)?\b', r'\bsuspend(?:ed)?\b', r'\baction required\b', r'\bwithin 24 hours\b']),
            ("Credential Harvest", [r'\bpassword\b', r'\blogin\b', r'\bcredentials?\b', r'\bverify your account\b', r'\bupdate your profile\b']),
            ("Financial Lure", [r'\binvoice\b', r'\bwire transfer\b', r'\brefund\b', r'\bbitcoin\b', r'\bbank account\b']),
            ("High-Risk Host", [r'https?://(?:\d{1,3}\.){3}\d{1,3}', r'https?://[^\s]+\.(?:xyz|top|pw|ru|buzz)'])
        ]
        
        for cat_name, patterns in categories:
            for pat in patterns:
                for match in re.finditer(pat, text, re.IGNORECASE):
                    start = max(0, match.start() - 25)
                    end = min(len(text), match.end() + 25)
                    context = ("..." if start > 0 else "") + text[start:end].strip() + ("..." if end < len(text) else "")
                    snippets.append({
                        "category": cat_name,
                        "matched_term": match.group(0),
                        "context": context
                    })
        return snippets[:10]  # Limit to top 10 relevant matches


if __name__ == "__main__":
    analyzer = SecurityAnalyzer()
    raw_sample = """From: "PayPal Account Security" <service-notice@paypal-update-center.xyz>
Reply-To: security-recovery@mail-server-hacker.pw
Subject: URGENT: Your PayPal Account Has Been Suspended!
Date: Thu, 10 Oct 2026 09:30:00 +0000
Authentication-Results: spf=fail (sender IP 185.220.101.5) smtp.mailfrom=paypal-update-center.xyz

Dear Customer,
We detected an unauthorized login attempt from IP 192.168.1.50.
Your account is suspended. Verify credentials immediately at http://192.168.1.50/login.php or payment of $250.00 will be frozen within 24 hours.
"""
    parsed = analyzer.parse_eml_content(raw_sample)
    iocs = analyzer.extract_iocs(parsed["body_text"])
    sender = analyzer.check_sender_mismatch(parsed["from"], parsed["return_path"], parsed["reply_to"])
    auth = analyzer.parse_auth_headers(parsed["headers"])
    from src.feature_engineering import extract_cyber_features
    features = extract_cyber_features(parsed["body_text"])
    
    risk = analyzer.calculate_risk_score(
        ml_probability=0.96,
        cyber_features=features,
        iocs=iocs,
        sender_analysis=sender,
        auth_analysis=auth,
        attachments=parsed["attachments"]
    )
    print("Security Analysis Verdict:", risk["verdict"])
    print("Risk Score:", risk["score"], "/ 100")
    print("Sender Spoofing:", sender["mismatches"])
    print("Extracted IOCs:", iocs["defanged_urls"], iocs["defanged_ips"])
    print("Authentication Failures:", auth["details"])
