<template>
  <main class="page-test">
    <NewHeader/>
    <scrollbar>
      <div class="cont">
        <div class="row">
          <div class="col">
            <h3>Input</h3>
            <t-field>
              <t-input-new :list="list" />
            </t-field>
            <t-field>
              <t-input-new :list="list" small />
            </t-field>
            <t-field>
              <t-input-new :list="list" small disabled />
            </t-field>
          </div>
          <div class="col">
            <h3>Checkbox</h3>
            <t-field inline label="Inline">
              <t-checkbox-new :list="list" small disabled />
            </t-field>
            <t-field inline label="Inline">
              <t-checkbox-new :list="list" small />
            </t-field>
            <t-field label="Normal">
              <t-checkbox-new :list="list" />
            </t-field>
          </div>
          <div class="col">
            <h3>Select</h3>
            <t-field>
              <t-select-new :list="list" />
            </t-field>
            <t-field>
              <t-select-new :list="list" small />
            </t-field>
          </div>
        </div>
        <div class="row">
          <div class="col">
            <h3>Input number</h3>
            <t-field label="Disabled">
              <DInputNumber disabled @change="change" />
            </t-field>
            <t-field label="Error">
              <DInputNumber :error="errorNumber" @change="change" @focus="errorNumber = ''" />
            </t-field>
            <t-field label="Medium">
              <DInputNumber v-model="inputNumber" @change="change" />
            </t-field>
            <t-field label="Small">
              <DInputNumber small icon="search" @change="change" />
            </t-field>
          </div>
          <div class="col">
            <h3>Input text</h3>
            <t-field label="Disabled">
              <DInputText disabled @change="change" />
            </t-field>
            <t-field label="Error">
              <DInputText v-model="errorText" :error="errorText" @change="change" @focus="errorText = ''" />
            </t-field>
            <t-field label="Medium">
              <DInputText v-model="inputString" icon="search" @change="change" />
            </t-field>
            <t-field label="Small">
              <DInputText small icon="search" @change="change" />
            </t-field>
          </div>
          <div class="col">
            <h3>Select</h3>
            <t-field label="Medium">
              <DSelect icon="search" :list="list"/>
            </t-field>
          </div>
        </div>

        <div class="row">
          <div class="col">
            <h3>Button primary</h3>
            <DButton color="primary" disabled>Disabled</DButton>
            <DButton color="primary" direction="left">Left</DButton>
            <DButton color="primary" size="small">Small</DButton>
            <DButton color="primary" size="small" direction="left">Small left</DButton>
          </div>
          <div class="col">
            <h3>Button secondary</h3>
            <DButton color="secondary" disabled>Disabled</DButton>
            <DButton color="secondary" direction="left">Left</DButton>
            <DButton color="secondary" size="small">Small</DButton>
            <DButton color="secondary" size="small" direction="left">Small left</DButton>
          </div>
          <div class="col">
            <h3>Button clear</h3>
            <DButton disabled>Disabled clear</DButton>
            <DButton @click="click">Clear</DButton>
          </div>
        </div>
      </div>
    </scrollbar>
  </main>
</template>

<script>
export default {
  name: 'page-not-found',
  components: {
    DInputText: () => import("@/components/global/design/forms/components/DInputText"),
    DInputNumber: () => import("@/components/global/design/forms/components/DInputNumber"),
    DSelect: () => import("@/components/global/design/forms/components/DSelect"),
    DButton: () => import("@/components/global/design/forms/components/DButton"),
    NewHeader: () => import("@/components/app/NewHeader"),
},
  data: () => ({
    inputNumber: 0,
    inputString: 'Текст',
    errorNumber: 'WTF!',
    errorText: 'Ops',
    list: [
      { value: 'test1', label: 'test1' },
      { value: 'test2', label: 'test2' },
      { value: 'test3', label: 'test3' },
    ],
  }),
  computed: {
    url() {
      return decodeURI(window.location.pathname);
    },
  },
  created() {
    console.log('sdsdsds');
    // window.location.href = "/";
  },
  methods: {
    change(e) {
      console.log(e);
    },
    click(value) {
      alert(value);
    },
  },
};
</script>

<style lang="scss" scoped>
.page-test {
  width: 100%;
  height: 100%;
  background-color: var(--color__bg);
  overflow: hidden;
}
.cont {
  height: 100%;
  width: 100%;
  padding: 20px;
  background-color: var(--color__page);
  .row {
    width: 100%;
    display: flex;
    margin: 40px 0 0 0;
  }
  .col {
    padding: 10px;
    height: 100%;
    flex: 1 1 33.3%;
  }
}
h3 {
  margin: 0 0 20px 0;
  text-decoration-line: underline;
  color: #c7dddd;
}
</style>