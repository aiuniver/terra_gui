# vue rules

По наименованию файла компонентов (PascalCase), названию компонента kebab-case, имена переменных (если это необходимо) через camalCase

```js
BlockBody.vue

export default {
  name: "block-body",
  data: () => ({
    title: "text", 
    listBlock: []
  }),
```

По именовании имена классов пишем по БЭМ с префиксом t, модификаторы через двойное тире --
t-block
t-block__header
t-block__body--hide

стили изолировано через препроцессор SASS (SCSS) 

<style lang="scss" scoped>
.t-block {
  &__header {

  }
  &__body {
    &—hide {
           
    }
  }
}
</style>

