<template>
  <div class="datasets">
    <p class="datasets-type flex align-center">
      <span class="mr-4">{{ list[selectedType] }}</span>
      <d-svg name="file-blank-outline" v-if="selectedType === 1" />
    </p>

    <div class="datasets-filter">
      <t-field label="">
        <d-input-text v-model="search" small />
      </t-field>

      <div class="datasets-filter-sort">
        <div class="datasets-filter-header">
          <div class="flex align-center" @click="show = !show">
            <span>{{ selectedSort.title }}</span>
            <d-svg name="arrow-chevron-down" />
          </div>
          <div class="datasets-filter-dropdown" v-if="show">
            <div v-for="(item, idx) in options" :key="idx" @click="onSelect(item)">
              {{ item.title }}
            </div>
          </div>
        </div>

        <div class="datasets-filter-display">
          <d-svg name="grid-cube-outline" :class="['ci-tile mr-4', { 'ci-tile--selected': display }]" @click.native="display = true" />
          <d-svg name="lines-justyfy" :class="['ci-tile', { 'ci-tile--selected': !display }]" @click.native="display = false" />
        </div>
      </div>
    </div>

    <scrollbar style="justify-self: stretch">
      <div v-if="display" class="datasets-cards">
        <DatasetCard @click.native="selectDataset(item)" v-for="(item, idx) in sortedList" :key="idx" :dataset="item" @click="onChoice" />
      </div>
      <Table v-else :selectedType="selectedType" />
      <div class="datasets__empty" v-if="!sortedList.length">Не найдено</div>
    </scrollbar>
  </div>
</template>

<script>
import DatasetCard from '@/components/datasets/choice/DatasetCard';
import Table from '@/components/datasets/choice/Table';
import { mapActions, mapGetters } from 'vuex';
export default {
  name: 'Datasets',
  props: ['datasets', 'selectedType'],
  components: {
    DatasetCard,
    Table,
  },
  data: () => ({
    list: ['Недавние датасеты', 'Проектные датасеты', 'Датасеты Terra'],
    search: '',
    display: true,
    sortIdx: 4,
    show: false,
    selectedSort: {
      title: 'По алфавиту от А до Я',
      value: 'alphabet',
      idx: 0,
    },
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
  }),
  computed: {
    ...mapGetters({
      project: 'projects/getProject',
    }),
    sortedList() {
      const sortDatasets = this.datasets || [];
      const search = this.search;
      return sortDatasets
        .sort((a, b) => a.name.localeCompare(b.name))
        .filter(i => {
          return search ? i.name.toLowerCase().includes(search.toLowerCase()) : true;
        });
    },
  },
  methods: {
    ...mapActions({
      choice: 'datasets/choice',
    }),
    isLoaded(dataset) {
      // console.log(this.project?.dataset?.alias)
      return this.project?.dataset?.alias === dataset.alias;
    },
    selectDataset(item) {
      this.$store.dispatch('datasets/selectDataset', item);
    },
    handleChangeFilter(sort) {
      this.selectedSort = sort;
    },
    randomDate(start, end, startHour, endHour) {
      let date = new Date(+start + Math.random() * (end - start));
      let hour = (startHour + Math.random() * (endHour - startHour)) | 0;
      date.setHours(hour);
      return date;
    },
    onSelect(value) {
      this.show = false;
      this.selectedSort = value;
    },
    onChoice(dataset) {
      this.$Modal
        .confirm({
          title: 'Загрузить',
          content: `Загрузить датасет ${dataset.name}?`,
        })
        .then(res => {
          if (res) {
            console.log(dataset);
            if (!dataset.training_available) return;
            if (!this.isLoaded(dataset)) {
              this.choice(dataset);
              // this.$store.dispatch('datasets/setSelectedIndex', key);
              return;
            }
          }
        });
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/variables/default.scss";
.datasets {
  padding: 30px 30px;
  padding: 30px 30px 0 30px;
  width: 100%;
  height: 100%;
  max-height: 865px;
  display: flex;
  flex-direction: column;
  gap: 30px;
  &-cards {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 20px;
  }
  &-type {
    font-size: 14px;
    font-weight: 600;
    height: 20px;
  }
  &__empty {
    text-align: center;
  }
}
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