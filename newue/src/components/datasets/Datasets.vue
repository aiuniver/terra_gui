<template>
  <div class="datasets">
    <p class="datasets-type flex align-center">
      <span class="mr-4">{{ list[selectedType] }}</span>
      <SvgContainer name="file-blank-outline" v-show="selectedType === 1" />
    </p>

    <div class="datasets-filter">
      <TField>
        <DSelect small :list="sortedListForSelect" />
      </TField>

      <div class="datasets-filter__sort">
        <div class="datasets-filter__sort-options" v-show="cardsDisplay">
          <div class="datasets-filter__sort-options--selected" @click="showSort(!showSortOpt)">
            <span>{{ selectedSort.title }}</span>
            <SvgContainer name="arrow-chevron-down" />
          </div>

          <div class="datasets-filter__sort-dropdown" v-show="showSortOpt">
            <div v-for="(item, idx) in getSortOptions" :key="idx" @click="selectSort(item.idx)">
              {{ item.title }}
            </div>
          </div>
        </div>

        <template v-if="selectedType !== 2">
          <SvgContainer
            name="grid-cube-outline"
            @click.native="cardsDisplay = true"
            :class="['ci-tile', { 'ci-tile--selected': cardsDisplay }]"
          />
          <SvgContainer
            name="lines-justyfy"
            @click.native="cardsDisplay = false"
            :class="['ci-tile', { 'ci-tile--selected': !cardsDisplay }]"
          />
        </template>
      </div>
    </div>

    <scrollbar style="justify-self: stretch">
      <div v-if="cardsDisplay" class="datasets-cards">
        <DatasetCard v-for="(item, idx) in sortedList" :key="idx" :dataset="item" />
      </div>
      <div v-else class="datasets-table__wrapper">
        <table class="datasets-table">
          <thead>
            <tr>
              <th v-for="(item, idx) in getHeaders" :key="idx" @click="selectTableSort(item.idx)">
                <span>{{ item.title }}</span>
                <i
                  v-show="selectedHeader === item.idx"
                  :class="['ci-icon', `ci-thin_long_02_${reverseSort ? 'down' : 'up'}`]"
                />
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, idx) in sortedTable" :key="'table' + idx">
              <td>
                <i class="ci-icon ci-image" />
                <span>{{ item.name }}</span>
				
              </td>
              <td>
                {{ item.size ? item.size.short.toFixed(2) + ' ' + item.size.unit : 'Предустановленный' }}
              </td>
              <td>1 минуту назад</td>
              <td>{{ item.date ? item.date.toLocaleString() : '' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </scrollbar>
  </div>
</template>

<script>
export default {
  name: 'Datasets',
  props: ['datasets', 'selectedType'],
  components: {
    DatasetCard: () => import('@/components/datasets/components/DatasetCard.vue'),
    SvgContainer: () => import('@/components/app/SvgContainer.vue'),
  },
  data: () => ({
    list: ['Недавние датасеты', 'Проектные датасеты', 'Датасеты Terra'],
    cardsDisplay: true,
    showSortOpt: false,
    sortIdx: 0,
    sortOptions: [
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
    theaders: [
      {
        title: 'Название',
        idx: 0,
      },
      {
        title: 'Размер',
        idx: 1,
      },
      {
        title: 'Автор',
        idx: 2,
      },
      {
        title: 'Последнее использование',
        idx: 3,
      },
      {
        title: 'Создание',
        idx: 4,
      },
    ],
    selectedHeader: 0,
    reverseSort: false,
  }),
  computed: {
    datasetList() {
      return this.datasets.filter(item => {
        if (this.selectedType === 1) return item.group === 'custom';
        if (this.selectedType === 2) return item.group !== 'custom';
      });
    },
	sortedListForSelect(){
		return this.sortedList.map(item => {
			return {
				value: item.alias,
				label: item.name
			}
		})
	},
    /* eslint-disable */
    sortedList() {
      if (this.selectedSort.value === 'alphabet') 
	  	return this.datasetList.sort((a, b) => a.name.localeCompare(b.name));
      if (this.selectedSort.value === 'alphabet_reverse')
        return this.datasetList.sort((a, b) => b.name.localeCompare(a.name));
      return this.datasetList.sort((a, b) => b.date - a.date);
    },
    sortedTable() {
      const selectedSort = this.theaders.find(item => item.idx === this.selectedHeader);
      if (selectedSort.idx === 0)
        return this.reverseSort
          ? this.datasetList.sort((a, b) => b.name.localeCompare(a.name))
          : this.datasetList.sort((a, b) => a.name.localeCompare(b.name));
      if (selectedSort.idx === 1)
        return this.reverseSort
          ? this.datasetList.sort((a, b) => b.size.value - a.size.value)
          : this.datasetList.sort((a, b) => a.size.value - b.size.value);
      if (selectedSort.idx === 2)
        return this.reverseSort
          ? this.datasetList.sort((a, b) => a.name.localeCompare(b.name))
          : this.datasetList.sort((a, b) => b.name.localeCompare(a.name));
      if (selectedSort.idx === 3)
        return this.reverseSort
          ? this.datasetList.sort((a, b) => a.date - b.date)
          : this.datasetList.sort((a, b) => b.date - a.date);
      if (selectedSort.idx === 4)
        return this.reverseSort
          ? this.datasetList.sort((a, b) => a.date - b.date)
          : this.datasetList.sort((a, b) => b.date - a.date);
    },
    /* eslint-enable */
    selectedSort() {
      return this.sortOptions.find(opt => opt.idx === this.sortIdx);
    },
    getSortOptions() {
      if (this.selectedType === 0) return this.sortOptions.slice(0, 4);
      if (this.selectedType === 1) return this.sortOptions.slice(0, 3);
      return this.sortOptions.slice(4, 6);
    },
    getHeaders() {
      const arr = [...this.theaders];
      if (this.selectedType === 1) {
        arr.splice(2, 1);
        return arr;
      }
      return arr.slice(0, 4);
    },
  },
  methods: {
    showSort(val = false) {
      this.showSortOpt = val;
    },
    selectSort(idx) {
      this.sortIdx = idx;
      this.showSortOpt = false;
    },
    selectTableSort(idx) {
      if (this.selectedHeader === idx) this.reverseSort = !this.reverseSort;
      else this.selectedHeader = idx;
    },
  },
  watch: {
    selectedType(idx) {
      if (idx === 2) {
        this.cardsDisplay = true;
        this.sortIdx = 4;
      } else this.sortIdx = 0;
    },
  },
};
</script>

<style lang="scss" scoped>
.datasets {
  padding: 30px 0 0 40px;
  width: 100%;
  height: 100%;
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
  &-filter {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-right: 60px;

    &__sort {
      display: flex;
      gap: 20px;
      align-items: center;
      &-dropdown {
        position: absolute;
        top: calc(100% + 10px);
        background-color: #242f3d;
        z-index: 1;
        border-radius: 4px;
        overflow: hidden;
        width: 100%;
        div {
          padding: 10px;
          &:hover {
            color: #65b9f4;
            background-color: #1e2734;
          }
        }
      }
      &-options {
        color: #a7bed3;
        font-size: 14px;
        position: relative;
        cursor: pointer;
        min-width: 220px;
        user-select: none;
        &--selected {
          display: flex;
          align-items: center;
          gap: 10px;
          justify-content: flex-end;
        }
      }
    }
  }
  &-table {
    font-size: 14px;
    font-weight: 400;
    width: 100%;
    &__wrapper {
      width: calc(100% - 150px);
      position: relative;
    }
    tr > *:nth-child(2) {
      text-align: right;
    }
    thead {
      background-color: #17212b;
      color: #6c7883;
      position: sticky;
      top: 0;
      i {
        color: #65b9f4;
        font-size: 20px;
        vertical-align: middle;
      }
      tr {
        height: 35px;
      }
      th {
        font-weight: inherit;
        padding: 0 50px;
        min-width: 150px;
        user-select: none;
        &:first-child {
          padding: 15px 10px;
        }
        * {
          vertical-align: middle;
          cursor: pointer;
        }
      }
    }
    tbody {
      tr {
        height: 55px;
        cursor: pointer;
        &:hover {
          background-color: #0e1621;
        }
      }
      td {
        color: #f2f5fa;
        padding: 15px 50px;
        white-space: nowrap;
        text-overflow: ellipsis;
        overflow: hidden;
        max-width: 450px;
        &:first-child {
          padding: 15px 10px;
        }
        i {
          font-size: 19px;
          color: #6c7883;
          margin-right: 15px;
        }
        * {
          vertical-align: middle;
        }
      }
    }
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