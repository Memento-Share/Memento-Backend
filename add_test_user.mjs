// scripts/add_test_user.mjs
// Usage:
//    SUPABASE_URL="https://dspaaqbthcdwtxfflovn.supabase.co" \
//   SUPABASE_ANON_KEY="YOUR_ANON_KEY" \
//   node scripts/add_test_user.mjs

const SUPABASE_URL = "https://dspaaqbthcdwtxfflovn.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzcGFhcWJ0aGNkd3R4ZmZsb3ZuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIzMTg2NTQsImV4cCI6MjA4Nzg5NDY1NH0.Atf2EAxi7BkqAgV3mrs_SWaqJP61iNzCCAwa213nckQ";


const endpoint = `${SUPABASE_URL.replace(/\/$/, "")}/rest/v1/rpc/create_user`;

const payload = {
  p_username: "rpc_test_user",
  p_phone: `555-${Math.floor(1000 + Math.random() * 9000)}`,
};

const res = await fetch(endpoint, {
  method: "POST",
  headers: {
    apikey: SUPABASE_ANON_KEY,
    Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify(payload), // ✅ REQUIRED
});

const text = await res.text();

if (!res.ok) {
  console.error("RPC failed:", res.status, res.statusText);
  console.error(text);
  process.exit(1);
}

console.log("RPC result:", text);