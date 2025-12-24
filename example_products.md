# PRODUCT SPECIFICATIONS FOR TEST SCENARIO GENERATION

---

### PRODUCT 1: SplitMate (Consumer FinTech)

**Overview:**
SplitMate is a mobile application designed for groups to track shared expenses and settle debts. It solves the problem of "who owes who" after group trips or shared living situations.

**Core Scope:**

- Group management (create groups, invite via link).
- Expense logging with support for multi-currency.
- Debt simplification algorithm (reducing total transaction count).
- Settlement integration via deep links to 3rd party payment apps.
- Offline support.

**Functional Logic & Constraints:**

1. **Expense Entry:** Users must be able to split bills equally, by percentages, by specific shares (e.g., 2 shares for User A, 1 for User B), or by exact adjustment amounts. The total split must equal the expense amount; if not, the system must block the save.
2. **Currency Handling:** The group has a "Base Currency." Expenses entered in foreign currencies (e.g., JPY) are converted to the Base Currency (e.g., USD) using the exchange rate at the time of entry. Users can manually override the exchange rate.
3. **Debt Simplification:** The system acts as a ledger. It does not hold money. It must calculate the net balance. If A owes B $10, and B owes C $10, the system should suggest A pays C $10 directly.
4. **Offline Mode:** Users can create expenses without internet. These queue locally. When connectivity returns, the app syncs. If a sync conflict occurs (two users edited the same expense), the latest timestamp wins (Last-Write-Wins strategy).

**User Flows:**

- **Trip Setup:** Creating a group, setting the default currency, and inviting contacts.
- **The "Complex Dinner":** One user pays the bill, but the split is unequal (User A had alcohol, User B didn't).
- **Settlement:** A user checks their balance, clicks "Settle", selects a payment method (Venmo), and marks the debt as paid. The recipient must acknowledge the payment for the debt to clear from the ledger.

---

### PRODUCT 2: GlucoGuard (IoT & Health)

**Overview:**
GlucoGuard is a smartphone companion app for the "G-Pump 3000" insulin pump. It serves as a secondary display for glucose data and a remote control for insulin delivery. Safety is the highest priority.

**Core Scope:**

- Real-time Bluetooth Low Energy (BLE) pairing with the pump.
- Visualization of Continuous Glucose Monitor (CGM) data.
- Remote bolus (insulin delivery) capability.
- Critical alerting system.

**Functional Logic & Constraints:**

1. **Connectivity:** The app maintains a persistent BLE connection. If the connection drops for >15 minutes, a local notification must trigger. Upon reconnection, the app must request "backfill data" from the pump's internal memory to fill gaps in the graph.
2. **Bolus Calculator:** The user inputs "Carbohydrates (grams)." The app calculates the insulin unit dose based on the user's pre-configured "Insulin-to-Carb Ratio" and "Correction Factor."
3. **Safety Limits:** The app must strictly enforce a "Max Bolus" limit (hardcoded to 20 units). If a calculated dose exceeds this, the app must cap it and warn the user.
4. **Critical Alerts:** If glucose readings fall below 55 mg/dL (Hypoglycemia), the app must play a loud alarm sound, overriding the phone's "Silent" or "Do Not Disturb" modes (using critical entitlement APIs).

**User Flows:**

- **Pairing:** Initial handshake between phone and pump requiring a 6-digit PIN displayed on the pump hardware.
- **Meal Entry:** User inputs carbs, reviews the calculated dose, authenticates via Biometrics (FaceID/Fingerprint), and sends the command.
- **Emergency:** Handling the sequence when the pump detects a rapid drop in glucose levels.

---

### PRODUCT 3: TaskFlow Pro (SaaS Productivity)

**Overview:**
A web-based Kanban project management tool targeting enterprise software teams. The key differentiator is strict Role-Based Access Control (RBAC) and state-transition rules to prevent workflow violations.

**Core Scope:**

- Kanban Board (Columns: To Do, In Progress, Review, Done).
- Card management (Title, Description, Assignee, Labels).
- User Roles: Admin, Editor, Viewer.
- Workflow Enforcement Rules.

**Functional Logic & Constraints:**

1. **RBAC:**
   - "Viewers" can only read cards and comment. They cannot move cards or edit descriptions.
   - "Editors" can move cards and edit content but cannot delete boards.
   - "Admins" have full access.
2. **State Transitions:** A card cannot be moved to the "Done" column unless the "Code Review" checklist item is marked as complete. If a user tries to drag it there without this criteria, the card snaps back to its previous column and an error toast appears.
3. **Concurrency:** The system uses optimistic locking. If User A and User B open the same card, and User A saves changes, User B's subsequent save attempt must fail with a "Content has changed" error, prompting them to refresh.

**User Flows:**

- **Onboarding:** Admin creates a board and invites a team with mixed roles.
- **The "Blocked" Move:** An Editor tries to move a ticket to production without meeting the required criteria (e.g., missing labels or unchecked boxes).
- **Audit:** An Admin reviewing the "Activity Log" to see who moved a specific card and when.

---

### PRODUCT 4: RouteMaster (Logistics & Delivery)

**Overview:**
A mobile application for last-mile delivery drivers. It handles route navigation, proof of delivery, and exception handling for restricted items.

**Core Scope:**

- Turn-by-turn navigation integration.
- Dynamic stop ordering.
- Proof of Delivery (Photo capture & Signature).
- Age Verification flows.

**Functional Logic & Constraints:**

1. **Geofencing:** The "Complete Delivery" button is disabled unless the driver's GPS coordinates are within a 50-meter radius of the delivery address.
2. **Age Restricted Items:** If a package contains alcohol (flagged in metadata), the app intercepts the delivery flow. It forces the driver to scan the recipient's ID. The app validates the ID's date of birth. If the recipient is under 21 or the ID is expired, the delivery cannot be completed.
3. **Exception Handling:** If a customer is not home, the driver selects "Unable to Deliver." The app prompts for a reason (No Access, Customer Unavailable, Dog) and automatically reschedules the package for the next day.

**User Flows:**

- **Start Shift:** Driver logs in, performs a vehicle safety checklist, and loads the route.
- **Restricted Delivery:** Delivering a package requiring ID verification, including the failure path (customer has no ID).
- **Navigation:** Handling a scenario where the driver deviates from the route or traffic requires a route recalculation.

---

### PRODUCT 5: WhisperNet (Social & Anonymity)

**Overview:**
An anonymous professional feedback platform where employees review managers. The system relies on cryptographic hashing to ensure the server never knows which user wrote which review, while still verifying employment.

**Core Scope:**

- Corporate Email Verification.
- Anonymous Posting.
- Aggregated Reporting Dashboards.
- Privacy Thresholds.

**Functional Logic & Constraints:**

1. **The Double-Blind Signup:** Users verify via work email (e.g., name@company.com). The system sends a magic link. Clicking the link creates a user account that is cryptographically unlinked from the email address.
2. **Privacy Threshold:** A manager cannot see feedback about themselves until a minimum of 3 unique users have submitted reviews. If only 1 or 2 reviews exist, the dashboard shows "Data Pending" to prevent the manager from guessing the author's identity.
3. **Content Sanitization:** Before submission, the text is analyzed for PII (Personally Identifiable Information). If the user types a phone number or a specific name in the body, the system warns them to remove it before allowing submission.

**User Flows:**

- **Verification:** The process of proving employment without attaching the email to the resulting profile.
- **The "Safe" Review:** A user writing a review and triggering the PII warning system.
- **Manager View:** A manager logging in to see their "Leadership Score," interacting with the logic that hides data when the sample size is too small.
