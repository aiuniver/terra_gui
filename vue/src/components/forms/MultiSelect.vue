<template>
  <div :class="['t-multi-select', { 't-inline': inline }]">
    <label class="t-multi-select__label">{{ label }}</label>
    <input
      class="t-multi-select__input"
      v-model="search"
      :name="name"
      :disabled="disabled"
      :placeholder="placeholder"
      @focus="focus"
      @blur="select(false)"
    />
    <div class="t-multi-select__content" v-show="show">
      <template v-for="(item, i) in filterList">
        <div class="t-multi__item" :key="i">
          <span class="t-multi-select__item--check"></span>
          <span class="t-multi-select__item--title">{{ item.label }}</span>
        </div>
      </template>
      <div v-if="!filterList.length" class="t-multi-select__item">Нет данных</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TMultiSelect',
  props: {
    name: String,
    lists: {
      type: Array,
      required: true,
      default: () => [],
    },
    placeholder: String,
    disabled: Boolean,
    label: {
      type: String,
      default: 'Label',
    },
    inline: Boolean,
    value: String,
  },
  data() {
    return {
      selected: {},
      show: false,
      search: '',
    };
  },
  created() {
    // this.$emit("selected", { name: this.value });
    this.search = this.value;
  },
  computed: {
    filterList() {
      return this.lists
        ? this.lists.filter(item => {
            const search = this.search;
            return search ? item.label.toLowerCase().includes(search.toLowerCase()) : true;
          })
        : [];
    },
  },
  methods: {
    select(item) {
      // console.log(item);
      if (item) {
        this.selected = item;
        this.show = false;
        this.search = item.label;
        this.$emit('input', this.selected.value);
        this.$emit('change', item);
      } else {
        this.search = this.selected.label;
        this.show = false;
      }
    },
    focus() {
      this.show = true;
      this.$emit('focus', true);
    },
  },
  watch: {
    value: {
      handler(value) {
        // console.log(value)
        this.show = false;
        this.search = value;
      },
    },
  },
};
</script>


<style lang="scss" scope>
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
    line-height: 1.25;
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
    padding: 0 10px;
    font-size: 0.875rem;
    font-weight: 400;
    width: 100%;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      border: 1px solid #e7ecf5;
    }
    &:disabled {
      border: 1px solid #6c7883;
      cursor: auto;
      opacity: 0.35;
    }
  }
  &__content {
    position: absolute;
    background-color: #242f3d;
    width: 100%;
    max-height: 190px;
    border: 1px solid #6c7883;
    box-shadow: 0px -8px 34px 0px rgb(0 0 0 / 5%);
    overflow: auto;
    border-radius: 0 0 4px 4px;
    z-index: 102;
    color: #a7bed3;
    font-size: 0.7em;
    line-height: 1em;
    padding: 8px;
    text-decoration: none;
    display: block;
  }
}
.t-multi__item {
  padding: 2px 12px;
  line-height: 1.5;
  text-align: left;
  cursor: pointer;
  &:hover {
    color: #e7ecf5;
    background-color: #6c7883;
  }
  &--check {
    position: absolute;
    top: 0;
    left: 0;
    height: 25px;
    width: 25px;
    background-color: #eee;
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  & label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
  }
  & input {
    height: 22px;
    font-size: 12px;
    line-height: 24px;
    width: 100px;
  }
  & .t-multi-select__content {

  }
}
</style>