SELECT
    d.id AS document_id,
    d.title,
    d.access_roles,
    c.id AS chunk_id,
    c.heading_path,
    LEFT(c.content, 500) AS content_preview
FROM document_chunks c
JOIN documents d
    ON d.id = c.document_id
WHERE
    c.content ILIKE '%measurable artifact%'
    OR c.content ILIKE '%48 hours%'
    OR c.content ILIKE '%missed meetings%';