SELECT
    id,
    tenant_id,
    title,
    source_type,
    access_roles,
    created_at
FROM documents
WHERE title ILIKE '%HyperOnboard%';