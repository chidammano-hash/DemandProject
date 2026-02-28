import type { ProductThemeId, ProductTheme } from "@/types/theme";
import { generalTheme } from "./general";

export const PRODUCT_THEMES: Record<ProductThemeId, ProductTheme> = {
  general: generalTheme,
};

export function getProductTheme(id: ProductThemeId): ProductTheme {
  return PRODUCT_THEMES[id];
}

export { generalTheme };
