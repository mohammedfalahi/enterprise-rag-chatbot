-- Get tenant ID
SELECT id, name
FROM public.tenants;

-- Get your test user's Auth ID
SELECT id, email
FROM auth.users
WHERE email = 'falahi@gmail.com';