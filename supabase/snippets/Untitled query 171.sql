SELECT
    COUNT(*) AS documents_without_chunks
FROM documents d
WHERE NOT EXISTS (
    SELECT 1
    FROM document_chunks dc
    WHERE dc.document_id = d.id
);