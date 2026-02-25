import type { ProductTheme, ProductThemeId } from "@/types/theme";
import { wineSpiritsTheme } from "./wineSpirits";
import { generalTheme } from "./general";
import { obsidianTheme } from "./obsidian";

export const PRODUCT_THEMES: Record<ProductThemeId, ProductTheme> = {
  "wine-spirits": wineSpiritsTheme,
  general: generalTheme,
  obsidian: obsidianTheme,
};

export const THEME_ORDER: ProductThemeId[] = ["wine-spirits", "general", "obsidian"];

export function getProductTheme(id: ProductThemeId): ProductTheme {
  return PRODUCT_THEMES[id];
}

export { wineSpiritsTheme, generalTheme, obsidianTheme };
