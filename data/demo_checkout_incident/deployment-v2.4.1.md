# Synthetic deployment record: checkout-api v2.4.1

All names, identifiers, and timestamps in this record are fabricated for the
IncidentIQ demonstration.

## Deployment window

- Release: `checkout-api v2.4.1`
- Started: `2025-02-18T09:55:00Z`
- 25% traffic: `2025-02-18T10:02:00Z`
- 100% traffic: `2025-02-18T10:06:00Z`
- Image rollback to `v2.4.0`: `2025-02-18T10:26:00Z`

## Intended changes

1. Add request-stage timing around cart, inventory, database, and payment calls.
2. Change payment retry backoff from fixed 250 ms to capped exponential backoff.
3. Consolidate database-pool settings into a shared deployment template.
4. Add a fallback path when the shipping-zone cache refresh is delayed.

## Configuration note

The shared template defaults `DB_POOL_MAX` to `4` for local and low-memory
canary environments. The production release checklist expected the existing
override of `40` to remain active. The deployment record confirms the expected
value but does not contain the rendered production configuration.

## Pre-deployment checks

- Unit and integration checks passed in the synthetic staging environment.
- A staging checkout completed against the payment-provider sandbox.
- Database migration status reported no pending schema changes.
- Checkout error rate was below 0.5% during the 15-minute baseline.

## Rollout observations

- Health checks remained green throughout rollout.
- No CPU or memory threshold alert fired during deployment.
- At `10:10Z`, the shipping-zone cache reported a delayed refresh. Fallback
  responses remained enabled, so this warning was considered non-blocking.
- At `10:18Z`, operators began investigating database waits and payment
  timeouts.
- Rolling the application image back at `10:26Z` did not immediately restore
  the checkout error rate. The shared runtime configuration was not rolled back
  with the image.

## Open questions recorded during the incident

- Was the expected production database-pool override present on every new pod?
- Did the payment-provider timeouts originate upstream or after local queueing?
- Did the new retry behavior amplify request concurrency?
- Was the shipping-cache warning related, or merely concurrent noise?
