import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

VERSIONS = ["1.0.0", "1.1.0", "1.2.0", "2.0.0", "2.1.0", "2.2.0"]
MODULES = [
    "Authentication", "Payment Gateway", "User Profile",
    "Search Engine", "Notification System", "Data Analytics",
    "API Integration", "File Upload", "Dashboard", "Reporting"
]
SEVERITIES = ["Low", "Medium", "High"]
STATUSES = ["Open", "Fixed", "Reopened"]

BUG_TEMPLATES = {
    "Authentication": [
        "Session token expires prematurely during active use",
        "Password reset email not sent to registered users",
        "OAuth2 callback URL mismatch causing login failure",
        "Two-factor authentication code not validated correctly",
        "Remember me functionality fails after browser restart",
        "Login page freezes on slow network connections",
        "LDAP integration returns incorrect user roles",
        "JWT token not invalidated on logout",
        "Brute force protection blocks legitimate users",
        "Single sign-on redirect loop in Firefox",
    ],
    "Payment Gateway": [
        "Duplicate charges on retry after network timeout",
        "Currency conversion rounding error in checkout",
        "Refund processing fails for orders older than 90 days",
        "Payment confirmation email delayed by over 10 minutes",
        "Card validation fails for valid Amex cards",
        "Stripe webhook not received for failed payments",
        "Subscription renewal does not apply discount codes",
        "Invoice PDF generated with incorrect tax amounts",
        "PayPal integration breaks on sandbox environment",
        "3D Secure authentication not triggered for high-value orders",
    ],
    "User Profile": [
        "Profile picture upload fails for images larger than 2MB",
        "Username change not reflected across all pages",
        "Email update requires re-verification on each login",
        "Address fields accept invalid postal codes",
        "Profile deletion does not remove associated data",
        "Notification preferences reset after profile save",
        "Timezone setting ignored in activity timestamps",
        "Bio field strips HTML entities incorrectly",
        "Account deactivation still allows API access",
        "Profile completeness percentage calculated incorrectly",
    ],
    "Search Engine": [
        "Search returns no results for exact product name match",
        "Filters not applied when navigating back from detail page",
        "Autocomplete suggestions include deleted items",
        "Search indexing lag causes stale results up to 30 minutes",
        "Boolean operators AND/OR not supported in query string",
        "Fuzzy search returns irrelevant results for short terms",
        "Category filter resets when sorting is changed",
        "Pagination breaks when search query contains special characters",
        "Search results not updated after product price change",
        "Typo correction overrides intentional user input",
    ],
    "Notification System": [
        "Push notifications not delivered on iOS 16 devices",
        "Email digest sends duplicate notifications for same event",
        "In-app notification badge count not reset after reading",
        "SMS notifications sent to unverified phone numbers",
        "Notification preferences not respected for critical alerts",
        "Bulk notification job times out for user base over 50k",
        "Unsubscribe link in emails leads to 404 page",
        "Notification template variables not replaced in subject line",
        "Real-time notifications delayed by 5 minutes during peak hours",
        "Notification history not paginated correctly",
    ],
    "Data Analytics": [
        "Dashboard metrics show incorrect date range on load",
        "Export to CSV truncates rows beyond 10000 records",
        "Custom report builder crashes on nested group-by",
        "Real-time chart not updated when new data arrives",
        "Funnel visualization missing intermediate steps",
        "Cohort analysis calculation off by one day",
        "Metric comparison period calculation uses wrong formula",
        "Data refresh spinner never stops on error state",
        "Heatmap colors inverted for negative values",
        "Scheduled report delivery fails on weekends",
    ],
    "API Integration": [
        "Rate limiting headers not returned in API responses",
        "Webhook signature verification fails intermittently",
        "API versioning not enforced for deprecated endpoints",
        "CORS headers missing for cross-origin requests",
        "Pagination cursor invalidated after record deletion",
        "API timeout set to 30s but client receives 10s error",
        "Batch API endpoint fails silently on partial success",
        "Authentication token refresh race condition causes 401",
        "API response schema changes break older client versions",
        "GraphQL mutation returns 200 on database constraint error",
    ],
    "File Upload": [
        "Large file upload fails silently without error message",
        "Virus scanning blocks legitimate PDF documents",
        "File type validation bypassed with double extension trick",
        "Upload progress bar resets to zero on network interruption",
        "Concurrent uploads from same user cause file corruption",
        "Thumbnail generation fails for HEIC image format",
        "Uploaded files not accessible after S3 bucket policy change",
        "File rename on upload strips special characters incorrectly",
        "Download speed throttled unintentionally for free tier users",
        "File preview not shown for Excel files with macros",
    ],
    "Dashboard": [
        "Widget layout not saved across browser sessions",
        "Date picker defaults to wrong timezone for UTC users",
        "Drag and drop widget rearrangement breaks on mobile",
        "Dashboard loads blank on first visit after cache clear",
        "Filter panel collapsed state not persisted",
        "KPI cards show NaN when denominator is zero",
        "Print view cuts off right side of charts",
        "Dashboard sharing link expires too quickly",
        "Embedded dashboard iframe resizes incorrectly",
        "Dark mode toggle resets to light on page refresh",
    ],
    "Reporting": [
        "Monthly report includes data from previous month",
        "PDF report generation fails for reports over 50 pages",
        "Report scheduling UI does not show next run time",
        "Column sorting in table report resets on export",
        "Report template variables not escaped in output",
        "Comparative report percentages exceed 100% due to rounding",
        "Scheduled report sent to deactivated email accounts",
        "Report access control does not inherit parent folder permissions",
        "Chart annotations not included in exported images",
        "Report query timeout causes partial data to be displayed",
    ],
}

VERSION_RELEASE_DATES = {
    "1.0.0": datetime(2023, 1, 15),
    "1.1.0": datetime(2023, 4, 10),
    "1.2.0": datetime(2023, 7, 20),
    "2.0.0": datetime(2023, 11, 5),
    "2.1.0": datetime(2024, 2, 28),
    "2.2.0": datetime(2024, 6, 15),
}

MODULE_SEVERITY_BIAS = {
    "Authentication": [0.15, 0.35, 0.50],
    "Payment Gateway": [0.10, 0.30, 0.60],
    "User Profile": [0.30, 0.45, 0.25],
    "Search Engine": [0.25, 0.50, 0.25],
    "Notification System": [0.35, 0.45, 0.20],
    "Data Analytics": [0.30, 0.45, 0.25],
    "API Integration": [0.20, 0.40, 0.40],
    "File Upload": [0.25, 0.45, 0.30],
    "Dashboard": [0.40, 0.45, 0.15],
    "Reporting": [0.30, 0.50, 0.20],
}

MODULE_REOPEN_RATE = {
    "Authentication": 0.30,
    "Payment Gateway": 0.25,
    "User Profile": 0.15,
    "Search Engine": 0.20,
    "Notification System": 0.18,
    "Data Analytics": 0.22,
    "API Integration": 0.28,
    "File Upload": 0.16,
    "Dashboard": 0.12,
    "Reporting": 0.14,
}

rows = []
bug_counter = 1000

for version in VERSIONS:
    release_date = VERSION_RELEASE_DATES[version]
    num_bugs = random.randint(55, 90)

    for _ in range(num_bugs):
        module = random.choice(MODULES)
        severity_probs = MODULE_SEVERITY_BIAS[module]
        severity = np.random.choice(SEVERITIES, p=severity_probs)

        reopen_prob = MODULE_REOPEN_RATE[module]
        fixed_prob = 1 - reopen_prob - 0.10
        open_prob = 0.10
        status = np.random.choice(STATUSES, p=[open_prob, fixed_prob, reopen_prob])

        template_bugs = BUG_TEMPLATES[module]
        description = random.choice(template_bugs)

        severity_occurrence_map = {"Low": (1, 5), "Medium": (3, 15), "High": (8, 40)}
        occ_range = severity_occurrence_map[severity]
        occurrences = random.randint(*occ_range)

        severity_fix_map = {"Low": (1, 7), "Medium": (3, 14), "High": (7, 30)}
        fix_range = severity_fix_map[severity]
        time_to_fix = round(random.uniform(*fix_range), 1) if status == "Fixed" else round(random.uniform(fix_range[1], fix_range[1] * 2), 1)

        days_offset = random.randint(0, 90)
        report_date = release_date + timedelta(days=days_offset)

        bug_id = f"BUG-{bug_counter}"
        bug_counter += 1

        rows.append({
            "Bug ID": bug_id,
            "App Version": version,
            "Module": module,
            "Bug Description": description,
            "Severity": severity,
            "Status": status,
            "Occurrences": occurrences,
            "Time to Fix (days)": time_to_fix,
            "Release Date": release_date.strftime("%Y-%m-%d"),
            "Report Date": report_date.strftime("%Y-%m-%d"),
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("bugs_data.csv", index=False)
print(f"Dataset generated: {len(df)} rows, {df.shape[1]} columns")
print(df.head())
print(df["Severity"].value_counts())
print(df["Status"].value_counts())
print(df["Module"].value_counts())
