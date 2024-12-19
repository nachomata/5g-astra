// @ts-check
import { defineConfig } from 'astro/config';

import tailwind from '@astrojs/tailwind';

import preact from '@astrojs/preact';

// https://astro.build/config
export default defineConfig({
    devToolbar: {
        enabled: false
    },
    output: "server",
    integrations: [tailwind(), preact()]
});