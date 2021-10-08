<template>
  <div :class="['t-multi-select', { 't-inline': inline }]" v-click-outside="outside">
    <label class="t-multi-select__label">
      <slot>{{ label }}</slot>
    </label>
    <div
      :class="['t-multi-select__input', { 't-multi-select__error': error }, { 't-multi-select__input--show': show }]"
    >
      <span
        :class="['t-multi-select__input--text', { 't-multi-select__input--active': input }]"
        :title="input"
        @click="click"
      >
        {{ input || placeholder }}
      </span>
    </div>
    <div class="t-multi-select__content" v-show="show">
      <scrollbar style="height: 165px">
        <div class="t-milti-select__inner">
          <div v-if="filterList.length" class="t-multi__item" @click="select(checkAll)">
            <span :class="['t-multi__item--check', { 't-multi__item--active': checkAll }]" />
            <span class="t-multi__item--title">Выбрать все</span>
          </div>
          <template v-for="(item, i) in filterList">
            <div class="t-multi__item" :key="i" :title="item.label" @click="select(item)">
              <span :class="['t-multi__item--check', { 't-multi__item--active': active(item) }]"></span>
              <span class="t-multi__item--title">{{ item.label }}</span>
            </div>
          </template>
          <div v-if="!filterList.length" class="t-multi__item t-multi__item--empty">
            <span class="t-multi__item--title">Нет данных</span>
          </div>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-multi-select-table',
  props: {
    name: String,
    id: Number,
    label: {
      type: String,
      default: 'Label',
    },
    placeholder: {
      type: String,
      default: 'Не выбрано',
    },
    disabled: Boolean,
    inline: Boolean,
    value: Array,
    table: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    // selected: [],
    show: false,
    pagination: 0,
  }),
  computed: {
    handlers: {
      set(value) {
        this.$store.dispatch('tables/setHandlers', value);
      },
      get() {
        return this.$store.getters['tables/getHandlers'];
      },
    },
    selected: {
      set(value) {
        this.$store.dispatch('tables/setSaveCols', { id: this.id, value });
      },
      get() {
        return this.$store.getters['tables/getSaveCols'](this.id);
      },
    },
    errors() {
      return this.$store.getters['datasets/getErrors'](this.id);
    },
    error() {
      const key = this.name;
      return this.errors?.[key]?.[0] || this.errors?.parameters?.[key]?.[0] || '';
    },
    input() {
      const labels = this.selected.map(item => item.label);
      const name = this.filterList.filter(item => labels.includes(item.label)).map(item => item.label);
      // this.clear(name);
      return name.join();
    },
    checkAll() {
      return this.filterList.length === this.selected.length;
    },
    cols() {
      return [].concat(
        ...this.table.map(item => {
          // console.log(item);
          const arr = item.table.map(td => {
            return { label: td[0], value: null, id: `${td[0]}[${item.label}]`, name: td[0], table: item.label };
          });
          return arr;
        })
      );
    },
    colsHandlers() {
      return [].concat(
        ...this.handlers.map(item => {
          let all = [];
          let arr = [];
          for (let key in item.table) {
            arr = arr.concat(
              ...item.table[key].map(td => {
                return {
                  label: `${td} (${item.name})`,
                  name: td,
                  value: item.id,
                  id: `${td}[${key}](${item.id})`,
                  table: key,
                };
                // return { label: `${td}[${key}](${item.name})`, value: [item.name] };
              })
            );
          }
          // console.log(arr);
          return all.concat(arr);
        })
      );
    },
    filterList() {
      return [].concat(...this.cols, ...this.colsHandlers).sort(function (a, b) {
        return a.label.toLowerCase() < b.label.toLowerCase() ? -1 : 1;
      });

      // .filter(item => !item.id || item.id === this.id)
      // .filter(item => filter.includes(item.type));
    },
  },
  methods: {
    clear(name) {
      this.selected = this.selected.filter(item => name.includes(item.label));
    },
    click() {
      this.show = true;
      if (this.error) {
        // console.log(this.id, this.name);
        this.$store.dispatch('datasets/cleanError', { id: this.id, name: this.name });
      }
    },
    active({ id }) {
      return !!this.selected.find(item => item.id === id);
    },
    outside() {
      if (this.show) {
        this.show = false;
      }
    },
    select(list) {
      // console.log(list);
      if (typeof list === 'boolean') {
        this.selected = this.filterList.map(item => (!list ? item : null)).filter(item => item);
      } else {
        if (this.selected.find(item => item.id === list.id)) {
          this.selected = this.selected.filter(item => item.id !== list.id);
          // this.handlers = this.handlers.map(item => {
          //   if (item.id === list.id) {
          //     item.layer = 0;
          //   }
          //   return item;
          // });
        } else {
          this.selected = [...this.selected, list];
          // this.handlers = this.handlers.map(item => {
          //   if (item.id === list.id) {
          //     item.layer = this.id;
          //   }
          //   return item;
          // });
        }
      }
      // console.log(this.selected);
      // this.$emit('change', { name: 'cols_names', value: this.selected });
      // this.$emit('multiselect', { value: this.selected, id: this.id });
      // this.mixinCheck(this.selected, this.id);
    },
  },
  // created() {
  //   // console.log(this.value);
  //   // console.log(this.filterList.filter(item => item));
  //   const value = this.value;
  //   if (Array.isArray(value)) {
  //     this.selected = this.filterList.filter(item => value.includes(item.value));
  //   }
  // },
  created() {
    console.log('created');
  },
  mounted() {
    console.log('moudsd');
  },
  watch: {
    // filterList() {
    //   this.selected = this.selected.filter(element => this.colsHandlers.find(item => item.id === element.id));
    // },
    // selected(value) {
    // console.warn('value')
    // this.$emit('change', { name: 'cols_names', value });
    // },
  },
};
</script>


<style lang="scss" scoped>
.t-multi-select {
  position: relative;
  margin-bottom: 10px;
  &__label {
    width: 150px;
    max-width: 130px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    border: 1px solid #6c7883;
    border-radius: 4px;
    background: #242f3d;
    height: 42px;
    padding: 0;
    font-size: 0.875rem;
    font-weight: 400;
    width: 100%;
    display: flex;
    align-items: center;
    &--show {
      border-bottom-right-radius: 0px;
      border-bottom-left-radius: 0px;
    }
    &--text {
      flex-grow: 1;
      text-overflow: ellipsis;
      white-space: nowrap;
      overflow: hidden;
      color: #747b82;
      font-size: 0.75rem;
      padding: 0 5px;
    }
    &--active {
      color: #fff;
    }
    i {
      opacity: 0;
      transition: opacity 0.3s ease-in-out;
      cursor: pointer;
    }
    &:hover i {
      opacity: 1;
    }
  }
  &__error {
    border-color: #b53b3b;
  }
  &__content {
    padding: 2px 0;
    position: absolute;
    background-color: #242f3d;
    width: 100%;
    max-height: 190px;
    border: 1px solid #6c7883;
    box-shadow: 0px -8px 34px 0px rgb(0 0 0 / 5%);
    overflow: auto;
    border-radius: 0 0 4px 4px;
    z-index: 102;
    overflow: auto;
  }
}
.t-multi__item {
  display: flex;
  padding: 1px 20px 0 5px;
  align-items: center;
  cursor: pointer;
  &--empty {
    cursor: default;
  }
  &:hover {
    color: #e7ecf5;
    background-color: #6c7883;
  }
  &--empty:hover {
    background-color: #242f3d;
  }
  &--check {
    height: 10px;
    background-color: #eee;
    margin-right: 5px;
    border-radius: 2px;
    flex: 0 0 10px;
  }
  &--active {
    background-color: #5191f2;
    width: 10px;
    height: 10px;
    border: 1px solid white;
  }
  &--title {
    color: #a7bed3;
    font-size: 0.7em;
    line-height: 1.5;
    text-align: left;
    cursor: pointer;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
    flex: 1 0 auto;
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  & .t-multi-select__label {
    width: auto;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    font-size: 0.75rem;
  }
  & .t-multi-select__input {
    height: 24px;
    font-size: 12px;
    width: 100px;
  }
  & .t-multi-select__content {
    width: auto;
    top: 23px;
    min-width: 100px;
  }
}
</style>