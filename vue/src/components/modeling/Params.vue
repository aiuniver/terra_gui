<template>
  <div class="params">
    <Navbar />
    <scrollbar>
      <div class="params__items">
        <div class="params__items--item">
          <Input
            v-model="block.name"
            :label="'Название слоя'"
            :type="'text'"
            :parse="'name'"
            :name="'name'"
            :disabled="isBlock"
            @change="saveModel"
          />
          <Autocomplete2
            :value="block.typeLabel"
            :list="listWithoutOutputInput"
            label="Тип слоя"
            name="type"
            :disabled="isBlock || isInput"
            @change="changeType"
          />
          <template v-for="({ name, label, parse, list }, i) of datatypes">
            <t-field :label="label" :key="'datatype' + i">
              <t-select-new :value="block.id"  :list="list" :parse="parse" :name="name" @change="changeId({ ...$event, id: block.id })"/>
            </t-field>
          </template>
        </div>
        <at-collapse :value="collapse">
          <at-collapse-item v-show="main.items.length" class="mb-3" title="Параметры слоя">
            <Forms :data="main" :id="block.id" @change="change" />
          </at-collapse-item>
          <at-collapse-item v-show="extra.items.length" class="mb-3" title="Дополнительные параметры">
            <Forms :data="extra" :id="block.id" @change="change" />
          </at-collapse-item>
          <at-collapse-item v-show="!isBlock" class="mb-3" title="Размерность слоя" notChange>
            <Shape
              v-if="block.shape && block.shape.input"
              v-model="block.shape.input"
              :label="'Размерность входных данных'"
              :name="'shape_input'"
              :disabled="block.type !== 'Input' || !!project.dataset"
              @change="saveModel"
            />
            <Shape
              v-if="block.shape && block.shape.output"
              :value="block.shape.output"
              :label="'Размерность выходных данных'"
              :name="'shape_output'"
              :disabled="true"
            />
          </at-collapse-item>
        </at-collapse>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import Input from '@/components/forms/Input.vue';
import Navbar from '@/components/modeling/comp/Navbar.vue';
import Shape from '@/components/forms/Shape.vue';
import Autocomplete2 from '@/components/forms/Autocomplete2.vue';
import Forms from '@/components/modeling/comp/Forms.vue';
import { mapGetters } from 'vuex';
// import serialize from "@/assets/js/serialize";

// import Select from "@/components/forms/Select.vue";
export default {
  name: 'Params',
  components: {
    Shape,
    Autocomplete2,
    Forms,
    Navbar,
    Input,
  },
  data: () => ({
    collapse: ['0', '2'],
    oldBlock: null,
  }),
  computed: {
    ...mapGetters({
      list: 'modeling/getList',
      layers: 'modeling/getLayersType',
      layersForm: 'modeling/getLayersForm',
      buttons: 'modeling/getButtons',
      block: 'modeling/getBlock',
      project: 'projects/getProject',
    }),
    datatypes() {
      return this.layersForm.filter(({ name }) => name === `datatype_${this.block.group}`)
    },
    isBlock() {
      return !this.block.id;
    },
    isInput() {
      return this.block.group === 'input';
    },
    listWithoutOutputInput() {
      return this.list.filter(item => !(item.value.toLowerCase() === 'input'));
    },

    buttonSave() {
      return this.buttons?.save || false;
    },
    main() {
      const blockType = this.block?.type;
      if (Object.keys(this.layers).length && blockType) {
        const items = this.layers[blockType]?.main || [];
        const value = this.block?.parameters?.main || {};
        return { type: 'main', items, value, blockType };
      } else {
        return { type: 'main', items: [], value: {} };
      }
    },
    extra() {
      const blockType = this.block?.type;
      if (Object.keys(this.layers).length && blockType) {
        const items = this.layers[blockType]?.extra || [];
        const value = this.block?.parameters?.extra || {};
        return { type: 'extra', items, value, blockType };
      } else {
        return { type: 'extra', items: [], value: {} };
      }
    },
  },
  methods: {
    async changeId(value) {
      await this.$store.dispatch('modeling/changeId', value);
    },
    async saveModel() {
      await this.$store.dispatch('modeling/updateModel', this.block);
    },
    async changeType({ value }) {
      await this.$store.dispatch('modeling/typeBlock', { type: value, block: this.block });
    },
    async change({ type, name, value }) {
      // console.group();
      console.log({ type, name, value });
      // console.log(this.collapse);
      // console.groupEnd();
      if (this.block.parameters) {
        this.block.parameters[type][name] = value;
      } else {
        this.oldBlock.parameters[type][name] = value;
      }
      this.$emit('change');
      this.saveModel();
    },
  },
  watch: {
    block: {
      handler(newBlock, oldBlock) {
        this.oldBlock = oldBlock;
        // this.$store.dispatch('modeling/setSelect', newBlock?.id);
        console.log(newBlock, oldBlock);
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  width: 400px;
  flex-shrink: 0;
  border-left: #0e1621 solid 1px;
  overflow: hidden;
  height: 100%;
  // border-left: #0e1621  1px solid;
  &__items {
    height: 100%;
    padding-bottom: 20px;
    &--item {
      padding: 20px;
    }
  }
}

.params-actions {
  padding: 20px 10px;
}
</style>
