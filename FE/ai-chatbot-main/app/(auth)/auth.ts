import { compare } from 'bcrypt-ts';
import NextAuth, { type DefaultSession } from 'next-auth';
import type { DefaultJWT } from 'next-auth/jwt';
import Credentials from 'next-auth/providers/credentials';
import { config } from 'dotenv';
import { resolve } from 'path';
import { readFileSync } from 'fs';
import { DUMMY_PASSWORD } from '@/lib/constants';
import { createGuestUser, getUser } from '@/lib/db/queries';
import { authConfig } from './auth.config';

// Load .env.local explicitly for NextAuth initialization
const envPath = resolve(process.cwd(), '.env.local');
const envResult = config({ path: envPath });

// Fallback: Read file directly if dotenv fails
let authSecret = process.env.AUTH_SECRET || process.env.NEXTAUTH_SECRET;

if (!authSecret) {
  try {
    // Try reading as UTF-8 first, then UTF-16 LE if that fails
    let envFile: string;
    try {
      envFile = readFileSync(envPath, 'utf-8');
    } catch {
      // If UTF-8 fails, try UTF-16 LE (Windows default)
      const buffer = readFileSync(envPath);
      envFile = buffer.toString('utf16le');
    }

    // Remove BOM (Byte Order Mark) if present
    if (envFile.charCodeAt(0) === 0xfeff) {
      envFile = envFile.slice(1);
    }

    // Remove any null bytes (UTF-16 artifacts)
    envFile = envFile.replace(/\u0000/g, '');

    // Remove any non-printable characters at the start
    envFile = envFile.replace(/^[\uFEFF\u200B-\u200D\u2060]+/, '');

    console.log('[AUTH] Reading .env.local file directly...');
    console.log('[AUTH] File length:', envFile.length);
    console.log('[AUTH] First 100 chars:', JSON.stringify(envFile.substring(0, 100)));

    // Split by lines and find AUTH_SECRET
    const lines = envFile.split(/\r?\n/);
    console.log('[AUTH] Total lines:', lines.length);

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith('AUTH_SECRET=')) {
        authSecret = line.substring('AUTH_SECRET='.length).trim();
        // Remove quotes if present
        authSecret = authSecret.replace(/^["']|["']$/g, '');
        process.env.AUTH_SECRET = authSecret;
        console.log(
          '[AUTH] ✅ Loaded AUTH_SECRET from line',
          i + 1,
          ':',
          authSecret.substring(0, 10) + '...'
        );
        break;
      }
    }

    // Fallback: Try regex patterns if line-by-line didn't work
    if (!authSecret) {
      const patterns = [
        /AUTH_SECRET\s*=\s*([^\s#\r\n]+)/m, // Most flexible
        /^AUTH_SECRET=(.+)$/m, // Standard format
        /^AUTH_SECRET\s*=\s*(.+)$/m, // With spaces
      ];

      for (const pattern of patterns) {
        const match = envFile.match(pattern);
        if (match && match[1]) {
          authSecret = match[1].trim();
          authSecret = authSecret.replace(/^["']|["']$/g, '');
          process.env.AUTH_SECRET = authSecret;
          console.log(
            '[AUTH] ✅ Loaded AUTH_SECRET via regex:',
            authSecret.substring(0, 10) + '...'
          );
          break;
        }
      }
    }

    if (!authSecret) {
      console.error('[AUTH] ❌ Could not find AUTH_SECRET in .env.local file');
      console.error(
        '[AUTH] Lines containing AUTH:',
        lines.filter(l => l.includes('AUTH'))
      );
    }
  } catch (error: any) {
    console.error('[AUTH] Failed to read .env.local:', error.message);
    console.error('[AUTH] Error stack:', error.stack);
  }
}

if (!authSecret) {
  console.error('[AUTH] Missing AUTH_SECRET environment variable');
  console.error('[AUTH] Env file path:', envPath);
  console.error('[AUTH] Dotenv result:', envResult);
  console.error(
    '[AUTH] Current env vars:',
    Object.keys(process.env).filter(k => k.includes('AUTH') || k.includes('SECRET'))
  );
}

export type UserType = 'guest' | 'regular';

declare module 'next-auth' {
  interface Session extends DefaultSession {
    user: {
      id: string;
      type: UserType;
    } & DefaultSession['user'];
  }

  // biome-ignore lint/nursery/useConsistentTypeDefinitions: "Required"
  interface User {
    id?: string;
    email?: string | null;
    type: UserType;
  }
}

declare module 'next-auth/jwt' {
  interface JWT extends DefaultJWT {
    id: string;
    type: UserType;
  }
}

export const {
  handlers: { GET, POST },
  auth,
  signIn,
  signOut,
} = NextAuth({
  ...authConfig,
  secret: authSecret,
  providers: [
    Credentials({
      credentials: {},
      async authorize({ email, password }: any) {
        const users = await getUser(email);

        if (users.length === 0) {
          await compare(password, DUMMY_PASSWORD);
          return null;
        }

        const [user] = users;

        if (!user.password) {
          await compare(password, DUMMY_PASSWORD);
          return null;
        }

        const passwordsMatch = await compare(password, user.password);

        if (!passwordsMatch) {
          return null;
        }

        return { ...user, type: 'regular' };
      },
    }),
    Credentials({
      id: 'guest',
      credentials: {},
      async authorize() {
        const [guestUser] = await createGuestUser();
        return { ...guestUser, type: 'guest' };
      },
    }),
  ],
  callbacks: {
    jwt({ token, user }) {
      if (user) {
        token.id = user.id as string;
        token.type = user.type;
      }

      return token;
    },
    session({ session, token }) {
      if (session.user) {
        session.user.id = token.id;
        session.user.type = token.type;
      }

      return session;
    },
  },
});
