<template>
  <div class="t-segmentation-manual">
    <t-input
      v-model="qty"
      label="Количество классов"
      type="number"
      name="classes"
      inline
      @change="change"
      @input="inputCheck"
    />
    <template v-for="(item, i) of +qty">
      <hr class="t-segmentation-manual__hr" :key="'hr_up' + i" />
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
      <Color :value="'#FFFFFF'" label="Цвет" :key="'classes_colors_' + i" inline />
      <hr v-if="+qty === i + 1" class="t-segmentation-manual__hr" :key="'hr_' + i" />
    </template>
  </div>
</template>

<script>
import Color from '../../forms/Color.vue';
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
      classes_colors: [],
    },
  }),
  computed: {},
  methods: {
    inputCheck(e) {
      if (+e > 99) this.qty = 99;
      if (+e < 0) this.qty = 0;
    },
    change(e) {
      console.log(this.model);
      if (this.isChange) {
        let value = e.target.value;
        value = this.type === 'number' ? +value : value;
        this.$emit('change', { name: this.name, value });
        this.isChange = false;
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-segmentation-manual {
  &__hr {
    height: 1px;
    border-width: 0;
    color: #17212b;
    background-color: #17212b;
    margin: 0 0 10px 0;
  }
}
</style>
