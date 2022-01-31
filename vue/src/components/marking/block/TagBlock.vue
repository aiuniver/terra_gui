<template>
  <div class="tag-block">
      <p class="tag-block__label">{{ title }}</p>
      <div class="tag-block__items">
          <div v-for="(val, i) in list" :key="i"
          class="tag-block__item"
          :style="{ backgroundColor: val.color }"
          >{{ val.name }}</div>
          <button class="tag-block__btn"><i></i> Добавить</button>
      </div>
  </div>
</template>

<script>
export default {
    name: 't-tag-block',
    data: () => ({
        
    }),
    props: ['title', 'list']
}
</script>

<style lang="scss" scoped>
.tag-block {
    &__label {
        font-size: 12px;
        line-height: 12px;
        color: #A7BED3;
    }
    &__items {
        margin-top: 10px;
        min-height: 100px;
        background: #242F3D;
        border: 1px solid #6C7883;
        border-radius: 4px;
        padding: 5px;
        display: flex;
        gap: 5px;
        flex-wrap: wrap;
    }
    &__item {
        height: 24px;
        display: inline-block;
        font-size: 12px;
        line-height: 24px;
        border-radius: 4px;
        padding: 0 8px;
    }
    &__btn {
        height: 24px;
        width: 95px;
        color: #A7BED3;
        border: 1px solid #6C7883;
        font-size: 12px;
        line-height: 24px;
        background: none;
        box-shadow: none;
        display: flex;
        justify-content: space-evenly;
        i {
            height: 24px;
            width: 24px;
            display: inline-block;
            background-size: contain;
            vertical-align: middle;
            background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE4IDEzSDEzVjE4QzEzIDE4LjU1IDEyLjU1IDE5IDEyIDE5QzExLjQ1IDE5IDExIDE4LjU1IDExIDE4VjEzSDZDNS40NSAxMyA1IDEyLjU1IDUgMTJDNSAxMS40NSA1LjQ1IDExIDYgMTFIMTFWNkMxMSA1LjQ1IDExLjQ1IDUgMTIgNUMxMi41NSA1IDEzIDUuNDUgMTMgNlYxMUgxOEMxOC41NSAxMSAxOSAxMS40NSAxOSAxMkMxOSAxMi41NSAxOC41NSAxMyAxOCAxM1oiIGZpbGw9IiNBN0JFRDMiLz4KPC9zdmc+Cg==');
        }
    }
}
</style>