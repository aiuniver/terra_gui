<template>
  <div class="t-table">
    <div class="t-table__header"></div>
    <div class="t-table__data">
      <div class="t-table__col" :style="{ padding: '1px 0' }">
        <div v-for="(i, idx) of 6" :key="'idx_r_' + i" class="t-table__row">{{ idx ? idx : '' }}</div>
      </div>
      <div class="t-table__border">
        <div
          v-for="(row, index) in origTable"
          :class="['t-table__col', { 't-table__col--active': row.active }]"
          :key="'row_' + index"
          @click="select(row, $event)"
        >
          <div v-if="row.active" class="t-table__select">
            <t-select-new small :list="[]"></t-select-new>
          </div>
          <div v-for="(item, i) in row" class="t-table__row" :key="'item_' + i">
            {{ item }}
          </div>
        </div>
      </div>
    </div>

    <div class="t-table__footer">
      <span>Список файлов</span>
      <div class="t-table__footer--btn" @click="show = true">
        <i class="t-icon icon-file-dot"></i>
      </div>
      <div v-show="show" class="t-table__dropdown">
        <div
          v-for="({ icon, event }, i) of items"
          :key="'icon' + i"
          class="t-table__dropdown--item"
          @click="$emit('event', { label, event }), (show = false)"
        >
          <i :class="['t-icon', icon]"></i>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CardTable',
  props: {
    label: String,
    type: String,
    id: Number,
    cover: String,
    table: Array,
  },
  data: () => ({
    show: false,
    items: [{ icon: 'icon-deploy-remove', event: 'remove' }],
    selected: [],
  }),
  computed: {
    getColor() {
      const id = this.group.filter(item => item.active);
      console.log();
      return { borderColor: this.color(id?.[0]?.layer) || '' };
    },
    origTable() {
      return this.originTable.map(item => {
        item.active = this.selected.includes(item[0]);
        return item;
      });
    },
    group: {
      set(value) {
        this.$store.dispatch('datasets/setTableGroup', value);
      },
      get() {
        return this.$store.getters['datasets/getTableGroup'];
      },
    },
  },
  created() {
    const newarr = [];
    this.table.forEach((el, index) => {
      el.forEach((elm, i) => {
        if (!newarr[i]) {
          newarr[i] = [];
        }
        newarr[i][index] = elm;
      });
    });
    console.log(newarr);
    this.originTable = newarr;
  },
  methods: {
    color(id) {
      const selectInputData = this.$store.getters['datasets/getInputDataByID'](id) || {};
      return selectInputData.color || '';
    },
    select([name], event) {
      console.log(event)
      if (this.selected.find(item => item === name)) {
        this.selected = this.selected.filter(item => item !== name);
      } else {
        // if () {

        // }
        this.selected.push(name);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-table {
  position: relative;
  font-size: 0.75rem;
  border-collapse: collapse;
  border: 1px solid #6c7883;
  border-radius: 8px;
  padding: 0 0 2px 0;
  display: flex;
  height: 152px;
  flex-direction: column;
  margin: 0 10px;
  &__header {
    display: flex;
    position: relative;
    padding: 0 0 0 23px;
    flex: 0 0 23px;
    user-select: none;
    // z-index: 2;
    background-color: #222c387d;
  }
  &__add {
    padding: 0 5px;
    width: 26px;
    border-bottom: none;
    display: flex;
    justify-content: flex-start;
    align-items: center;
    order: 999;
    i {
      width: 12px;
      height: 18px;
      cursor: pointer;
    }
  }
  &__label {
    // min-width: 92px;
    position: relative;
    border: 1px solid #6c7883;
    border-radius: 4px 4px 0 0;
    border-bottom: none;
    background-color: #17212b;
    padding: 2px 6px 2px 6px;
    margin-right: 1px;
    display: flex;
    justify-content: flex-start;
    align-items: center;
    cursor: default;
    margin-left: -2px;
    &:first-child {
      margin-left: 0px;
    }
    span {
      // margin-right: 5px;
    }
    i {
      width: 12px;
      height: 18px;
      cursor: pointer;
    }
    &--active {
      margin-left: 0;
      min-width: 92px;
      margin-bottom: -1px;
      margin-left: 0px;
      // order: -1 !important;
      justify-content: space-between;
    }
  }
  &__data {
    display: flex;
    user-select: none;
    // border: 1px solid #6c7883;
  }

  &__border {
    padding: 1px 0;
    display: flex;
    position: relative;
    // border-left: 1px solid #6c7883;
  }
  &__select {
    position: absolute;
    top: -24px;
    width: 100px;
  }
  &__col {
    display: flex;
    flex-direction: column;
    padding: 1px;
    &--active {
      min-width: 100px;
      padding: 0;
      border: 1px solid #6c7883;
      border-radius: 0px 0px 4px 4px;
    }
  }
  &__row {
    height: 17px;
    padding: 0 8px;
    text-overflow: ellipsis;
    overflow: hidden;
    &:nth-child(even) {
      background: #242f3d;
    }
    &:first-child {
      font-weight: bold;
      height: 19px;
    }
  }

  &__footer {
    flex: 0 0 24px;
    width: 100%;
    padding: 0px 8px;
    position: relative;
    display: flex;
    justify-content: space-between;
    // align-items: center;

    &--label {
      bottom: 0;
      border-radius: 0 0 3px 3px;
      padding: 4px 2px 2px 6px;
      text-overflow: ellipsis;
      overflow: hidden;
    }
    &--btn {
      padding: 0 6px 0 0;
      cursor: pointer;
      i {
        width: 16px;
      }
      &:hover {
      }
    }
  }
  &__dropdown {
    position: absolute;
    background-color: #2b5278;
    border-radius: 4px;
    right: 3px;
    bottom: 3px;
    z-index: 100;
    &--item {
      position: relative;
      width: 32px;
      height: 32px;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      &:hover {
        opacity: 0.7;
      }
      i {
        width: 14px;
      }
    }
  }
}
</style>
