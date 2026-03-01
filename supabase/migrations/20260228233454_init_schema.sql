-- =========================
-- USERS
-- =========================
create table public.users (
  id bigint generated always as identity primary key,
  username varchar not null,
  phone_num varchar not null
);

-- =========================
-- FAMILY
-- =========================
create table public.family (
  id bigint generated always as identity primary key,
  owner_user_id bigint not null references public.users(id) on delete cascade,
  viewer_user_id bigint not null references public.users(id) on delete cascade
);

-- =========================
-- MEDIA
-- =========================
create table public.media (
  id bigint generated always as identity primary key,
  photos varchar null,
  recordings varchar null,
  created_at timestamptz default now()
);

-- =========================
-- CONVOS
-- =========================
create table public.convos (
  id bigint generated always as identity primary key,
  owner_user_id bigint not null references public.users(id) on delete cascade,
  media_id bigint null references public.media(id) on delete set null,
  visibility varchar not null default 'private',
  publish_at timestamptz null,
  published_at timestamptz null,
  created_at timestamptz default now(),
  check (visibility in ('private','family'))
);

-- =========================
-- MESSAGES
-- =========================
create table public.messages (
  id bigint generated always as identity primary key,
  convo_id bigint not null references public.convos(id) on delete cascade,
  title varchar,
  date timestamptz,
  threads text not null,
  created_at timestamptz default now()
);