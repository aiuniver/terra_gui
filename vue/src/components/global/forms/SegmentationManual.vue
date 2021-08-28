<template>
  <div>
    <t-input
      v-model="qty"
      label="Количество классов"
      type="number"
      name="classes"
      inline
      @change="change"
    />
    <template v-for="item, i of +qty">
      <t-input
        :value="''"
        label="Название класса"
        type="text"
        name="classes_names"
        :key="'classes_names_' + i"
        :parse="'classes_names[]'"
        inline
        @change="change"
      />
      <Color :value="''" label="Цвет" :key="'classes_colors_' + i" inline />
    </template>
  </div>
</template>

<script>
import Color from '../../forms/Color.vue'
export default {
  name: 't-segmentation-manual',
  components: {
    Color,
  },
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    type: {
      type: String,
      default: 'text',
    },
    value: {
      type: [String, Number],
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
    small: Boolean,
    error: String,
  },
  data: () => ({
    qty: 0,
    loading: false,
    model: {
      classes_names: [],
      classes_colors: []
    }
  }),
  computed: {},
  methods: {
    change() {
      console.log(this.model)
      // if (this.isChange) {
      //   let value = e.target.value;
      //   value = this.type === 'number' ? +value : value;
      //   this.$emit('change', { name: this.name, value });
      //   this.isChange = false;
      // }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  // align-items: center;
  .t-field__label {
    padding: 6px 0 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
  }
  .t-field__button {
    flex: 0 0 100px;
    height: 24px;
    font-size: 12px;
    line-height: 24px;
  }
}
</style>
