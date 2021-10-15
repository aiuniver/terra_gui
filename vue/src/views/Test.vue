<template>
  <main class="page-not-found">
    <div class="cont">
      <!-- <div>
        <t-field inline>
          <t-input-new :list="list" small />
        </t-field>
        <t-field inline>
          <t-checkbox-new :list="list" small />
        </t-field>
        <t-field inline>
          <t-select-new :list="list" small />
        </t-field>
      </div> -->
      <div>
        <t-field>
          <DInput label="Error" @change="change" type="number" />
        </t-field>
        <t-field>
          <DInput v-model="input" label="Disabled" disabled @change="change"/>
        </t-field>
        <t-field label="Number, small">
          <DInput v-model="error" small :error="error" @change="change"/>
        </t-field>
        <t-field>
          <DInput v-model="input" icon="search" :error="error" @focus="error = ''" @change="change"/>
        </t-field>
      </div>
    </div>
  </main>
</template>

<script>
export default {
  name: 'page-not-found',
  components: {
    DInput: () => import('../components/global/design/forms/DInput.vue'),
  },
  data: () => ({
    input: 'Екатерина',
    error: 'fgfgfg',
    list: [
      { value: 'test', label: 'test' },
      { value: 'dfdfdfdfdfd', label: 'tedfdfdfddddddddfdfst' },
      { value: 'tesfd dfdft', label: 'te fdfdf dfdst' },
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
      console.log(e)
    }
  }
};
</script>

<style lang="scss" scoped>
.page-not-found {
  width: 100%;
  height: 100%;
  background-color: var(--color__bg);
  .cont {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background-color: var(--color__page);
    > div {
      width: 300px;
    }
  }
}
.title {
  color: #6c7883;
  font-size: 1.5rem;
  span {
    color: #a9a9a9;
    font-weight: 700;
  }
}
</style>