"use client";

import { useState } from "react";
import { supabase } from "../../lib/supabaseClient";
import { useRouter } from "next/navigation";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) alert(error.message);
    else router.push("/");
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 text-black">
      <form onSubmit={handleLogin} className="p-8 bg-white rounded-lg shadow-md w-96 space-y-4">
        <h2 className="text-2xl font-bold text-center text-gray-800">Login</h2>
        <div>
          <label className="block mb-1 text-sm font-medium text-gray-700">Email</label>
          <input 
            type="email" 
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full border rounded px-3 py-2 text-black"
            required 
          />
        </div>
        <div>
          <label className="block mb-1 text-sm font-medium text-gray-700">Password</label>
          <input 
            type="password" 
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border rounded px-3 py-2 text-black"
            required 
          />
        </div>
        <button type="submit" className="w-full bg-blue-600 text-white font-medium rounded py-2 hover:bg-blue-700">
          Sign In
        </button>
      </form>
    </div>
  );
}
