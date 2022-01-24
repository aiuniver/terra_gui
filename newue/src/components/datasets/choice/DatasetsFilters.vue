<template>
  <div class="datasets-filter">
    <TField>
      <DSelect small :list="sortedDatasets" />
    </TField>

    <div class="datasets-filter-sort">
      <div class="datasets-filter-header"  v-if="cardsDisplay">
        <div  class="flex align-center" @click="showSortOptions = !showSortOptions">
          <span>{{ selectedSort.title }}</span>
          <d-svg name="arrow-chevron-down" />
        </div>
        <div class="datasets-filter-dropdown" v-if="showSortOptions">
          <div v-for="(item, idx) in sortOptions" :key="idx" @click="selectSort(item.idx)">
            {{ item.title }}
          </div>
        </div>
      </div>

      <div class="datasets-filter-display" v-if="selectedType !== 2">
        <d-svg
          name="grid-cube-outline"
          @click.native="$emit('changeDisplay', true)"
          :class="['ci-tile mr-4', { 'ci-tile--selected': cardsDisplay }]"
        />
        <d-svg
          name="lines-justyfy"
          @click.native="$emit('changeDisplay', false), (showSortOptions = false)"
          :class="['ci-tile', { 'ci-tile--selected': !cardsDisplay }]"
        />
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DatasetsFilters',
  props: {
    data: {
      type: Array,
      default: () => [],
    },
    cardsDisplay: {
      type: Boolean,
      default: false,
    },
    selectedType: {
      type: [String, Number],
      default: 0,
    },
  },
  data: () => ({
    options: [
      {
        title: 'По алфавиту от А до Я',
        value: 'alphabet',
        idx: 0,
      },
      {
        title: 'По алфавиту от Я до А',
        value: 'alphabet_reverse',
        idx: 1,
      },
      {
        title: 'Последние созданные',
        value: 'last_created',
        idx: 2,
      },
      {
        title: 'Последние использованные',
        value: 'last_used',
        idx: 3,
      },
      {
        title: 'Популярные',
        value: 'popular',
        idx: 4,
      },
      {
        title: 'Последние добавленные',
        value: 'last_added',
        idx: 5,
      },
    ],
    sortIdx: 4,
    showSortOptions: false,
  }),
  methods: {
    selectSort(idx) {
      this.sortIdx = idx;
      this.showSortOptions = false;
      this.$emit('changeFilter', this.selectedSort);
    },
  },
  computed: {
    sortedDatasets() {
      return this.data.map(item => {
        return {
          value: item.alias,
          label: item.name,
        };
      });
    },
    sortOptions() {
      if (this.selectedType === 0) return this.options.slice(0, 4);
      if (this.selectedType === 1) return this.options.slice(0, 3);
      return this.options.slice(4, 6);
    },
    selectedSort() {
      return this.sortOptions.find(opt => opt.idx === this.sortIdx);
    },
  },
  watch: {
    selectedType(idx) {
      if (idx === 2) {
        this.$emit('changeDisplay', true);
        this.sortIdx = 4;
      } else this.sortIdx = 0;
    },
  },
};
</script>

<style lang="scss" scoped>
.datasets-filter {
  display: flex;
  justify-content: space-between;
  align-items: center;
  &-dropdown {
    position: absolute;
    top: calc(100% + 10px);
    right: 0;
    background-color: #242f3d;
    z-index: 1;
    border-radius: 4px;
    overflow: hidden;
    width: 220px;
    cursor: pointer;
    div {
      padding: 10px;
      &:hover {
        color: #65b9f4;
        background-color: #1e2734;
      }
    }
  }

  &-header {
    cursor: pointer;
    position: relative;
    user-select: none;
  }

  &-sort {
    display: flex;
    gap: 20px;
    position: relative;
    align-items: center;
  }
}

.ci-tile {
  display: inline-block;
  border-radius: 4px;
  &--selected {
    border: 1px solid $color-light-blue;
    &::v-deep svg {
      fill: $color-light-blue;
    }
  }
}
</style>