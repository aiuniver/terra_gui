<template>
  <div class="file-manager">
    <div class="file-manager-catalog" v-for="catalog in list" :key="JSON.stringify(catalog)">
      <div class="file-manager-catalog__header flex align-center mb-2" @click="handleClick(catalog.id)">
        <d-svg
          :name="active.includes(catalog.id) ? 'arrow-carret-right-active-fill' : 'arrow-carret-right-outline'"
          class="mr-1"
        />
        <d-svg name="folder" class="mr-1" />
        <span>{{ catalog.label }}</span>
      </div>
      <div class="file-manager-catalog__list ml-14" v-show="active.includes(catalog.id)">
        <div
          class="file-manager-catalog__list-item"
          @click="$emit('chooseFile', item)"
          v-for="item in catalog.list"
          :key="JSON.stringify(item)"
        >
          <BaseFileManager v-if="item.list" :list="item.list" />
          <div v-else class="flex align-center mb-2">
            <d-svg name="file-outline" class="mr-1" />
            <span>{{ item.label }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'FileManager',
  props: {
    list: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    active: [],
  }),
  methods: {
    handleClick(id) {
      if (this.active.includes(id)) {
        this.active.splice(
          this.active.findIndex(el => id === el),
          1
        );
      } else {
        this.active.push(id);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/variables/default.scss";
.file-manager {
  &-catalog {
    &__header {
      cursor: pointer;
    }
    &__list {
      &-item {
        cursor: pointer;
        transition: background 0.3s ease, fill 0.3s ease, color 0.3s ease;
        border-radius: 4px;

        &:hover {
          color: $color-light-blue;
          background: $color-dark-gray;
          &::v-deep svg {
            fill: $color-light-blue;
          }
        }
      }
    }
  }
}
</style>