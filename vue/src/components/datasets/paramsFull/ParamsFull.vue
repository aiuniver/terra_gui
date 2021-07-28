<template>
  <div class="params-full">
    <div class="params-full__inner">
      <div class="params-full__files">
        Lorem ipsum, dolor sit amet consectetur adipisicing elit. Dolorem ullam
        mollitia vitae incidunt nesciunt, laborum officiis accusantium. Adipisci
        quaerat quod laborum et tenetur? Et facilis incidunt aspernatur
        molestiae, laborum repellendus?
      </div>
      <div class="params-full__main">
        <div class="main__header">
          Lorem, ipsum dolor sit amet consectetur adipisicing elit. Iure magnam necessitatibus ipsa corporis impedit. Exercitationem animi deleniti voluptates vel ullam explicabo hic sequi nesciunt fuga sit reprehenderit, recusandae odio veniam?
        </div>
        <div class="main__center">
          <div class="main__center--left"></div>
          <div class="main__center--right"></div>
        </div>
        <div class="main__footer">
Lorem ipsum dolor sit amet consectetur adipisicing elit. Vero facere neque veniam dolore quia eum molestias dolorem, repudiandae maiores aut explicabo. Soluta ea facere delectus perferendis nam temporibus fuga id.
        </div>
      </div>
    </div>
  </div>
</template>

<script>
// import CardFormInput from "@/components/datasets/CardFormInput.vue";
// import CardFile from "@/components/datasets/CardFile.vue";

export default {
  name: "ParamsFull",
  components: {
    // CardFile,
  },
  data: () => ({}),
  computed: {
    // ...mapGetters({
    //   settings: "datasets/getSettings",
    // }),
  },
};
</script>


<style lang="scss">
.params-full__inner {
  width: 100%;
  display: -webkit-flex;
  display: flex;
}
.params-full {
  flex-shrink: 0;
  width: 100%;
  margin-left: 41px;
  flex: auto;
  display: -webkit-flex;
  display: flex;
  background-color: #17212b;
  &__files {
    flex: 0 0 190px;
    display: -webkit-flex;
    display: flex;
  }
  &__main {
    flex: 1 1 360px;
    display: -webkit-flex;
    display: flex;
    flex-direction: column;
    > .main__header{
      flex: 0 0 172px;
    }
    > .main__center{
      flex: 1 1;
      display: flex;
      &--left{
        
      }
      &--right{

      }

    }
    > .main__footer{
      flex: 0 0 96px;
    }
  }
}
</style>